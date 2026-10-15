#!/usr/bin/env python
"""Evaluate a checkpoint saved by train_bc.py, without training, at one or more execution
horizons (actions executed from each predicted chunk before the policy is called again).

Same evaluation as during training (train_bc.evaluate: --eval_num_envs simulators in
parallel, start states fixed by --eval_seed, the env of the run's dataset), so with the
run's own n_action_steps and the default seed it repeats the training-time evaluation
(up to the policy's sampling noise).

--layouts demo starts the episodes from the layouts of the recorded demos the policy was
trained on (robot, bottle, container, table; simulator i, round r: demo i + r * num_envs)
instead of random ones: high success there but not on random layouts points to
overfitting, low success on both to a mismatch between training and the simulator.

For each --n_action_steps value it writes, in the run folder:
  evals_offline/<checkpoint>_n<N>_<layouts>_seed<S>.mp4   every episode, as train_bc's evals/
  evals_offline/results.jsonl     one line per evaluation (appended), with the success rate
                                  per round (each simulator's 1st, 2nd... episode)
  evals_offline/episodes.jsonl    one line per episode (appended): simulator, round, demo
                                  (with --layouts demo), success, length

Usage: see resfit/lerobot/shell/eval_bc.sh.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from resfit.lerobot.policies.factory import get_policy_class  # noqa: E402
from resfit.lerobot.scripts.train_bc import evaluate  # noqa: E402
from resfit.pick_place.vec_env import create_vectorized_env  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run folder (outputs/resfit_bc/<run name>)")
    p.add_argument("--checkpoint", default="best",
                   help="best, latest, or a kept checkpoint, e.g. checkpoints/step_0040000 (train_bc's --save_every_eval)")
    p.add_argument("--n_action_steps", type=int, nargs="+", default=None,
                   help="execution horizons to evaluate, each <= the chunk size (default: the run's own)")
    p.add_argument("--episodes", type=int, default=50, help="episodes per evaluation")
    p.add_argument("--num_envs", type=int, default=25, help="simulators run in parallel")
    p.add_argument("--seed", type=int, default=1000, help="seed of the start states (train_bc's --eval_seed)")
    p.add_argument("--layouts", default="random", choices=["random", "demo"],
                   help="start layouts: random (as in training's evaluations) or the recorded demos'")
    # Task options of the env (resfit/pick_place/env.py), defaults: the original task
    p.add_argument("--terminate_on_fall", type=int, default=0, choices=[0, 1],
                   help="1: end an episode when the bottle falls (FELL in the video)")
    p.add_argument("--robot_start", default="demo", choices=["demo", "near_home"])
    p.add_argument("--robot_start_noise_deg", type=float, default=5.0)
    p.add_argument("--box_position", default="random", choices=["random", "fixed"])
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.episodes % args.num_envs:
        p.error("--episodes must be a multiple of --num_envs (same number of episodes per simulator)")
    return args


def main():
    args = parse_args()
    mp.set_start_method("spawn", force=True)
    run_dir = Path(args.run).resolve()
    run_cfg = json.loads((run_dir / "config.json").read_text())
    policy_dir = run_dir / args.checkpoint / "policy"
    policy_type = json.loads((policy_dir / "config.json").read_text())["type"]
    policy = get_policy_class(policy_type).from_pretrained(policy_dir)
    policy.to(args.device).eval()
    chunk_size = getattr(policy.config, "chunk_size", None) or policy.config.horizon
    horizons = args.n_action_steps or [policy.config.n_action_steps]
    if max(horizons) > chunk_size:
        raise SystemExit(f"--n_action_steps must be <= the chunk size ({chunk_size})")
    step = None
    if (run_dir / args.checkpoint / "eval.json").exists():
        step = json.loads((run_dir / args.checkpoint / "eval.json").read_text())["step"]
    logger.info(f"{run_dir.name}/{args.checkpoint} (step {step}): {policy_type}, chunk {chunk_size}, "
                f"execution horizons {horizons}, {args.episodes} episodes, {args.layouts} layouts, seed {args.seed}")

    out_dir = run_dir / "evals_offline"
    out_dir.mkdir(exist_ok=True)
    env_options = {"terminate_on_fall": bool(args.terminate_on_fall), "robot_start": args.robot_start,
                   "robot_start_noise_deg": args.robot_start_noise_deg, "box_position": args.box_position}
    logger.info(f"env options: {env_options}")
    env = create_vectorized_env(run_cfg["dataset"], num_envs=args.num_envs, device=args.device,
                                env_options=env_options)
    per_env = args.episodes // args.num_envs
    demo_queues = None
    if args.layouts == "demo":
        num_layouts = env.get_attr("num_layouts")[0]
        if args.episodes > num_layouts:
            raise SystemExit(f"--layouts demo: at most {num_layouts} episodes (one per demo)")
        demo_queues = [[i + r * args.num_envs for r in range(per_env)] for i in range(args.num_envs)]
    progress = tqdm(horizons, desc="horizons", unit="horizon", dynamic_ncols=True)
    try:
        with logging_redirect_tqdm():
            for n in progress:
                progress.set_postfix_str(f"n_action_steps={n}")
                episodes_bar = tqdm(total=args.episodes, desc=f"  n={n} episodes", unit="ep", leave=False,
                                    dynamic_ncols=True)
                # evaluate() calls policy.reset(), which rebuilds the action queues with this length
                policy.config.n_action_steps = n
                video = out_dir / f"{args.checkpoint.replace('/', '_')}_n{n}_{args.layouts}_seed{args.seed}.mp4"
                if demo_queues is not None:
                    env.set_attr("demo_queue", demo_queues)  # consumed by the resets: set again each time
                episode_log = []
                success_rate, mean_length, elapsed = evaluate(policy, env, args.episodes, args.seed, video, env.fps,
                                                             pbar=episodes_bar, episode_log=episode_log)
                episodes_bar.close()
                se = (success_rate * (1 - success_rate) / args.episodes) ** 0.5
                per_round = [sum(e["success"] for e in episode_log if e["round"] == r) / args.num_envs
                             for r in range(per_env)]
                fell_rate = sum(e["fell"] for e in episode_log) / args.episodes
                logger.info(f"n_action_steps {n:>3d}: success rate {100 * success_rate:.1f} % "
                            f"(± {100 * se:.1f}), fell {100 * fell_rate:.1f} %, "
                            f"per round {[f'{100 * x:.0f} %' for x in per_round]}, "
                            f"mean length {mean_length:.0f}, {elapsed:.0f} s -> {video.name}")
                time = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
                common = {"time": time, "checkpoint": args.checkpoint, "step": step, "n_action_steps": n,
                          "layouts": args.layouts, "seed": args.seed, "env_options": env_options}
                with open(out_dir / "results.jsonl", "a") as f:
                    f.write(json.dumps({**common, "episodes": args.episodes, "success_rate": success_rate,
                                        "success_rate_se": se, "fell_rate": fell_rate,
                                        "success_rate_per_round": per_round,
                                        "mean_episode_length": mean_length, "video": video.name}) + "\n")
                with open(out_dir / "episodes.jsonl", "a") as f:
                    for e in episode_log:
                        if demo_queues is not None:
                            e = {**e, "demo": demo_queues[e["env"]][e["round"]]}
                        f.write(json.dumps({**common, **e}) + "\n")
    finally:
        progress.close()
        env.close()


if __name__ == "__main__":
    main()
