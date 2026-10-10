#!/usr/bin/env python
"""Pre-train the base policy (behaviour cloning) on a pick-and-place LeRobot dataset.

Adapted from ResFiT's train_bc_dexmg.py for this task: the local LeRobot datasets of
resfit/lerobot/dataset/convert_pick_place_to_lerobot.py and the simulator of
resfit/pick_place/vec_env.py. The residual RL fine-tuning is a separate script that
starts from a checkpoint saved here.

Every --eval_freq steps the policy is evaluated in --eval_num_envs simulators in
parallel, over --eval_episodes episodes (each simulator runs the same number of
episodes, from start states fixed by --eval_seed, so every evaluation sees the same
episodes and their success rates can be compared). Then, in the run folder
(outputs/resfit_bc/<run name>/):
  latest/   always overwritten: policy/ (weights + config) and trainer_state.pt
            (step, optimizer, best success rate so far, W&B run id), to resume from
  best/     overwritten when the success rate beats the best so far: policy/ and
            eval.json (step, success rate); also uploaded to W&B as an artifact
            (run_<id>_best). This is the checkpoint the residual RL starts from.
  evals/    one video per evaluation, every episode with both cameras side by side
            (local only: no video is uploaded to W&B)
--resume <run folder> continues a run from its latest/ (same W&B run, same step,
optimizer state and best success rate).

W&B logging is required (wandb login once per machine).

Usage: see resfit/lerobot/shell/train_bc.sh.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
import torch
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.transforms import ImageTransforms, ImageTransformsConfig
from lerobot.common.datasets.utils import cycle
from lerobot.common.utils.random_utils import set_seed
from PIL import Image, ImageDraw, ImageFont
from termcolor import colored

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import wandb  # noqa: E402

from resfit.lerobot.dataset.cached_dataset import CachedVideoLeRobotDataset  # noqa: E402
from resfit.lerobot.policies.factory import get_policy_class, make_policy, make_policy_config  # noqa: E402
from resfit.pick_place.vec_env import create_vectorized_env  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
logger = logging.getLogger(__name__)

OUTPUTS = REPO / "outputs" / "resfit_bc"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=False, help="LeRobot dataset folder (videos/task/dataset_sim/lerobot/...)")
    p.add_argument("--policy", default="act", choices=["act"], help="policy architecture")
    p.add_argument("--chunk_size", type=int, default=16, help="actions predicted per policy call")
    p.add_argument("--n_action_steps", type=int, default=8, help="actions of each chunk executed before the next call")
    p.add_argument("--policy_kwargs", default="", help="other policy config overrides, as 'key=value key=value'")
    p.add_argument("--steps", type=int, default=100_000, help="total optimisation steps")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--grad_clip_norm", type=float, default=10.0)
    p.add_argument("--num_workers", type=int, default=32, help="data loading processes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_freq", type=int, default=100)
    p.add_argument("--eval_freq", type=int, default=10_000, help="evaluate (and save latest/, maybe best/) every N steps")
    p.add_argument("--eval_episodes", type=int, default=64, help="episodes per evaluation")
    p.add_argument("--eval_num_envs", type=int, default=16, help="simulators run in parallel during evaluation")
    p.add_argument("--eval_seed", type=int, default=1000, help="seed of the evaluation start states")
    p.add_argument("--run_name", default=None, help="run folder name (default: <date>_<policy>_<dataset>)")
    p.add_argument("--resume", default=None, help="run folder to continue (outputs/resfit_bc/<run name>)")
    p.add_argument("--wandb_project", default="pick-place-bc")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.resume is None and args.dataset is None:
        p.error("--dataset is required (unless --resume)")
    if args.eval_episodes % args.eval_num_envs:
        p.error("--eval_episodes must be a multiple of --eval_num_envs (same number of episodes per simulator)")
    return args


def parse_policy_kwargs(text):
    """'key=value key=value' -> dict, with ints, floats and booleans converted."""
    out = {}
    for token in text.replace(",", " ").split():
        key, _, value = token.partition("=")
        if not _:
            raise ValueError(f"--policy_kwargs: expected key=value, got '{token}'")
        if value.lower() in ("true", "false"):
            out[key] = value.lower() == "true"
        else:
            try:
                out[key] = int(value)
            except ValueError:
                try:
                    out[key] = float(value)
                except ValueError:
                    out[key] = value
    return out


# -----------------------------------------------------------------------------
# Checkpoints
# -----------------------------------------------------------------------------


def save_checkpoint(ckpt_dir: Path, policy, extra_files: dict):
    """Write policy/ and the given files to ckpt_dir, replacing it atomically (a crash
    while saving leaves the previous checkpoint intact)."""
    tmp = ckpt_dir.with_name(ckpt_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    policy.save_pretrained(tmp / "policy")
    for name, content in extra_files.items():
        if name.endswith(".pt"):
            torch.save(content, tmp / name)
        else:
            (tmp / name).write_text(json.dumps(content, indent=1))
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    tmp.rename(ckpt_dir)


def load_latest(run_dir: Path, policy, optimizer, device):
    """Load latest/ into the existing policy and optimizer (in place, so the optimizer
    keeps updating this policy's parameters). Returns the trainer state."""
    latest = run_dir / "latest"
    loaded = get_policy_class(policy.name).from_pretrained(latest / "policy")
    policy.load_state_dict(loaded.state_dict())
    policy.to(device)
    state = torch.load(latest / "trainer_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    return state


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def _font():
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except OSError:
        return ImageFont.load_default()


def _annotate(frame, text, success, font):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.text((4, 2), text, fill=(255, 255, 255), font=font)
    draw.text((4, 14), "SUCCESS" if success else "FAIL", fill=(0, 200, 0) if success else (220, 0, 0), font=font)
    return np.array(img)


def evaluate(policy, env, episodes, seed, video_path, fps):
    """Run `episodes` episodes, the same number in every simulator, from start states
    fixed by `seed`. Returns the success rate and the mean episode length; writes every
    counted episode to video_path."""
    num_envs = env.num_envs
    per_env = episodes // num_envs
    was_training = policy.training
    policy.eval()
    policy.reset()  # drop any action chunk left from the previous evaluation

    font = _font()
    writer = imageio.get_writer(video_path.as_posix(), fps=fps)
    obs, _ = env.reset(seed=seed)
    frames = [[f] for f in env.render()]
    done_count = np.zeros(num_envs, int)
    successes, lengths = 0, []
    t0 = time.perf_counter()
    while (done_count < per_env).any():
        with torch.inference_mode():
            action = policy.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        # Same-step autoreset: the frame of an env whose episode just ended already
        # shows its next episode's start; it opens that episode's frames.
        rendered = env.render()
        ended = (terminated | truncated).cpu().numpy()
        for i in range(num_envs):
            if not ended[i]:
                frames[i].append(rendered[i])
                continue
            if done_count[i] < per_env:
                success = bool(terminated[i]) and float(reward[i]) == 1.0
                successes += success
                lengths.append(len(frames[i]))
                n = int(done_count.sum()) + 1
                for f in frames[i]:
                    writer.append_data(_annotate(f, f"env {i} | episode {n}/{episodes}", success, font))
            done_count[i] += 1
            frames[i] = [rendered[i]]
        policy.reset(env_ids=[int(i) for i in np.flatnonzero(ended)])
    writer.close()
    if was_training:
        policy.train()
    elapsed = time.perf_counter() - t0
    logger.info(f"evaluation: {episodes} episodes in {elapsed:.0f} s")
    return successes / episodes, float(np.mean(lengths)), elapsed


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    args = parse_args()
    mp.set_start_method("spawn", force=True)
    device = torch.device(args.device)

    if args.resume:
        run_dir = Path(args.resume).resolve()
        saved = json.loads((run_dir / "config.json").read_text())
        # Resume with the run's own settings; only the step budget may be raised.
        for key, value in saved.items():
            if key not in ("resume", "steps"):
                setattr(args, key, value)
        args.resume = str(run_dir)
    else:
        dataset_name = Path(args.dataset).resolve().name
        args.run_name = args.run_name or f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.policy}_{dataset_name}"
        run_dir = OUTPUTS / args.run_name
        run_dir.mkdir(parents=True)
        args.dataset = str(Path(args.dataset).resolve())
        (run_dir / "config.json").write_text(json.dumps({k: v for k, v in vars(args).items() if k != "resume"},
                                                        indent=1))
    (run_dir / "evals").mkdir(exist_ok=True)
    set_seed(args.seed)

    # ---- dataset and policy ------------------------------------------------------
    dataset_dir = Path(args.dataset)
    repo_id = f"local/{dataset_dir.name}"
    ds_meta = LeRobotDatasetMetadata(repo_id, root=dataset_dir)
    policy_cfg = make_policy_config(args.policy, chunk_size=args.chunk_size, n_action_steps=args.n_action_steps,
                                    **parse_policy_kwargs(args.policy_kwargs))
    policy_cfg.device = device.type
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    image_transforms_cfg = ImageTransformsConfig(enable=True)
    # Frames and action chunks read from memory (see cached_dataset.py): same samples as
    # LeRobotDataset, without the per-sample video decoding that bottlenecked training.
    dataset = CachedVideoLeRobotDataset(repo_id, root=dataset_dir, delta_timestamps=delta_timestamps,
                                        image_transforms=ImageTransforms(image_transforms_cfg))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type != "cpu", drop_last=True, persistent_workers=args.num_workers > 0)
    dl_iter = cycle(dataloader)

    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    policy.train()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=policy_cfg.optimizer_lr,
                                  weight_decay=policy_cfg.optimizer_weight_decay)

    step, best_success_rate, best_step, wandb_id = 0, -1.0, None, None
    if args.resume:
        state = load_latest(run_dir, policy, optimizer, device)
        step, best_success_rate, best_step, wandb_id = (state["step"], state["best_success_rate"],
                                                         state["best_step"], state["wandb_run_id"])
        logger.info(colored(f"Resumed {run_dir.name} at step {step} (best success rate "
                            f"{best_success_rate:.2f} at step {best_step})", "cyan"))

    wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, name=args.run_name, dir=str(run_dir),
        id=wandb_id, resume="must" if wandb_id else None,
        config={**vars(args), "action_mode": ds_meta.info["action_mode"], "policy_config": asdict(policy_cfg),
                "delta_timestamps": delta_timestamps, "image_transforms": asdict(image_transforms_cfg),
                "dataset_episodes": ds_meta.total_episodes, "dataset_frames": ds_meta.total_frames},
    )
    logger.info(colored(f"Run folder: {run_dir} | W&B: {wandb.run.url}", "green"))

    eval_env = create_vectorized_env(dataset_dir, num_envs=args.eval_num_envs, device=device.type)

    def trainer_state(step):
        return {"step": step, "optimizer": optimizer.state_dict(), "best_success_rate": best_success_rate,
                "best_step": best_step, "wandb_run_id": wandb.run.id}

    # ---- training loop ----------------------------------------------------------------
    while step < args.steps:
        t0 = time.perf_counter()
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                 for k, v in next(dl_iter).items()}
        data_ms = (time.perf_counter() - t0) * 1000
        loss, loss_dict = policy.forward(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_freq == 0:
            iter_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[step {step:>7d}/{args.steps}] loss {loss.item():.4f} | data {data_ms:.0f} ms | "
                        f"iter {iter_ms:.0f} ms")
            wandb.log({"train/loss": loss.item(), "train/grad_norm": float(grad_norm),
                       **{f"train/{k}": v for k, v in (loss_dict or {}).items()},
                       "time/data_ms": data_ms, "time/iter_ms": iter_ms}, step=step)

        if step % args.eval_freq == 0 or step == args.steps:
            video_path = run_dir / "evals" / f"step_{step:07d}.mp4"
            success_rate, mean_length, eval_s = evaluate(policy, eval_env, args.eval_episodes, args.eval_seed,
                                                         video_path, eval_env.fps)
            improved = success_rate > best_success_rate
            if improved:
                best_success_rate, best_step = success_rate, step
            logger.info(colored(f"[step {step}] success rate {100 * success_rate:.1f} % "
                                f"(best {100 * best_success_rate:.1f} % at step {best_step})", "cyan"))
            wandb.log({"eval/success_rate": success_rate, "eval/mean_episode_length": mean_length,
                       "eval/best_success_rate": best_success_rate, "time/eval_s": eval_s}, step=step)

            save_checkpoint(run_dir / "latest", policy, {"trainer_state.pt": trainer_state(step)})
            if improved:
                save_checkpoint(run_dir / "best", policy, {"eval.json": {"step": step, "success_rate": success_rate}})
                artifact = wandb.Artifact(f"run_{wandb.run.id}_best", type="model",
                                          metadata={"step": step, "success_rate": success_rate})
                artifact.add_dir(str(run_dir / "best"))
                wandb.log_artifact(artifact, aliases=["best", f"step_{step}"])
                logger.info(colored(f"New best: saved {run_dir / 'best'}", "magenta"))

    eval_env.close()
    wandb.finish()
    logger.info(colored(f"Training finished. Best success rate {100 * best_success_rate:.1f} % at step "
                        f"{best_step}: {run_dir / 'best'}", "green", attrs=["bold"]))


if __name__ == "__main__":
    main()
