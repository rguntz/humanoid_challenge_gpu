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
            eval.json (step, success rate). This is the checkpoint the residual RL
            starts from. Checkpoints stay local: nothing is uploaded to W&B.
  evals/    one video per evaluation, every episode with both cameras side by side
            (local only: no video is uploaded to W&B)
--resume <run folder> continues a run from its latest/ (same W&B run, same step,
optimizer state and best success rate).

--normalizer per_dim_range normalizes the state and the actions per dimension to
[-1, 1], the formulation of the earlier JAX pipeline (resfit/pick_place/normalization.py);
the default, mean_std, is LeRobot's (mean and standard deviation). Images always use
mean_std.

Image augmentation (training only): LeRobot's colour jitter (3 of brightness, contrast,
saturation, hue, sharpness per sample), then, with --random_shift N > 0, a random shift
of up to N pixels (edge-padded by N, cropped back to the original size at a random
offset, independently per camera and per sample: DrQ's shift augmentation).

--precision bf16 (the default) runs the training forward pass in bfloat16 mixed
precision, with TF32 matmuls and the convolution weights in channels-last layout: ~1.9x
faster than fp32 on an RTX 3090 (115 vs 222 ms per step at batch 256). The weights,
gradients and optimizer stay in fp32, and evaluation runs in fp32.

Settings of the earlier JAX pipeline's plain BC (policy/train.py, agent=bc_vision_abs), each
selectable: --policy legacy_bc (its MLP head on Diffusion Policy's encoder, see
make_bc_policy_config), --cameras (it used agentview only), --n_obs_steps (To, observation
frames), --chunk_size (Ta, actions predicted) and --n_action_steps (execution horizon), --ema
(evaluate and save an exponential moving average of the weights, its warmup schedule: decay
1 - k^-0.75 capped at 0.9999, 0 for the first two updates), --optimizer legacy (AdamW betas
(0.95, 0.999), the policy's weight decay, learning rate warmed up linearly over 500 steps then
cosine-decayed to 0 at --steps) and --grad_clip_norm 0 (no clipping). With --ema, latest/policy
holds the trained weights (to resume) and best/policy the averaged ones (the evaluated policy).

W&B logging is required (wandb login once per machine).

Usage: see resfit/lerobot/shell/train_bc.sh.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import math
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
from torchvision.transforms import v2
from lerobot.configs.types import NormalizationMode
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
from resfit.pick_place.normalization import per_dim_range_stats  # noqa: E402
from resfit.pick_place.vec_env import create_vectorized_env  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
logger = logging.getLogger(__name__)

OUTPUTS = REPO / "outputs" / "resfit_bc"
CAMERA_FEATURES = {"agentview": "observation.images.agentview", "wrist": "observation.images.wrist"}
LEGACY_LR_WARMUP_STEPS = 500


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=False, help="LeRobot dataset folder (videos/task/dataset_sim/lerobot/...)")
    p.add_argument("--policy", default="act",
                   choices=["act", "mlp", "diffusion", "unet_bc", "flow_matching", "legacy_bc"],
                   help="act; mlp (ACT's vision, an MLP head: policies/mlp/); diffusion (Diffusion Policy's "
                        "Unet, denoising objective); unet_bc (the same Unet trained by plain behaviour "
                        "cloning); flow_matching (the same Unet, flow matching); legacy_bc (the earlier JAX "
                        "pipeline's plain BC): see make_bc_policy_config")
    p.add_argument("--chunk_size", type=int, default=16, help="Ta: actions predicted per policy call")
    p.add_argument("--n_action_steps", type=int, default=8,
                   help="execution horizon: actions of each chunk executed before the next call")
    p.add_argument("--n_obs_steps", type=int, default=None,
                   help="To: observation frames given to the policy (default: the policy's own; act and mlp: 1)")
    p.add_argument("--cameras", nargs="+", default=None, choices=list(CAMERA_FEATURES),
                   help="cameras given to the policy (default: all)")
    p.add_argument("--ema", type=int, default=0, choices=[0, 1],
                   help="evaluate and save an EMA of the weights (see the module docstring)")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "legacy"],
                   help="adamw: constant learning rate; legacy: see the module docstring")
    p.add_argument("--policy_kwargs", default="", help="other policy config overrides, as 'key=value key=value'")
    p.add_argument("--normalizer", default="mean_std", choices=["mean_std", "per_dim_range"],
                   help="state and action normalization (see the module docstring)")
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp32"],
                   help="training forward pass precision (see the module docstring)")
    p.add_argument("--random_shift", type=int, default=0,
                   help="random image shift augmentation, in pixels (0: off; see the module docstring)")
    p.add_argument("--steps", type=int, default=100_000, help="total optimisation steps")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--grad_clip_norm", type=float, default=10.0, help="0: no clipping")
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


def make_bc_policy_config(args, ds_meta):
    """The policy config of --policy, with --chunk_size, --n_action_steps and --policy_kwargs.
    diffusion, unet_bc and flow_matching are Diffusion Policy (policies/diffusion/) with objective
    "diffusion", "regression" or "flow_matching" (same Unet and observation encoder, see
    DiffusionConfig.objective; flow_matching samples with 10 Euler steps). For all three: the
    horizon is --chunk_size, the images are randomly cropped to 90 % of their size (Diffusion
    Policy's own shift augmentation; centre crop at evaluation) and the loss is masked on padded
    actions, as for ACT.
    legacy_bc is the plain BC of the earlier JAX pipeline (policy/xqc/agents/actors/bc/): the same
    observation encoder with objective "mlp" (an MLP 256-256 on the features of the --n_obs_steps
    frames, MSE loss), images randomly cropped to 116 x 116, and the chunk predicted from the
    current step (actions t .. t + Ta - 1, from frames t - To + 1 .. t).
    --n_obs_steps (To) overrides the policy's own number of observation frames."""
    kwargs = parse_policy_kwargs(args.policy_kwargs)
    if args.n_obs_steps is not None:
        kwargs = {"n_obs_steps": args.n_obs_steps, **kwargs}
    if args.policy in ("act", "mlp"):
        return make_policy_config(args.policy, chunk_size=args.chunk_size, n_action_steps=args.n_action_steps,
                                  **kwargs)
    h, w = ds_meta.features[ds_meta.camera_keys[0]]["shape"][:2]
    defaults = {"horizon": args.chunk_size, "n_action_steps": args.n_action_steps,
                "crop_shape": (int(0.9 * h), int(0.9 * w)), "do_mask_loss_for_padding": True,
                "objective": {"diffusion": "diffusion", "unet_bc": "regression", "flow_matching": "flow_matching",
                              "legacy_bc": "mlp"}[args.policy]}
    if args.policy == "flow_matching":
        defaults["num_inference_steps"] = 10  # Euler steps at sampling
    if args.policy == "legacy_bc":
        defaults.update(crop_shape=(min(116, h), min(116, w)), actions_from_current_step=True)
    return make_policy_config("diffusion", **{**defaults, **kwargs})


def ema_decay(step):
    """Decay of the weights' moving average after the step-th update (1-based): the earlier JAX
    pipeline's warmup schedule (policy/xqc/agents/common/ema.py, Diffusion Policy's EMAModel)."""
    k = step - 1
    return 0.0 if k <= 1 else min(0.9999, max(0.0, 1.0 - k ** -0.75))


def legacy_lr_lambda(total_steps):
    """Learning-rate factor per step: linear warmup over LEGACY_LR_WARMUP_STEPS from 0, then
    cosine decay to 0 at total_steps (optax.warmup_cosine_decay_schedule)."""
    def factor(step):
        if step < LEGACY_LR_WARMUP_STEPS:
            return step / LEGACY_LR_WARMUP_STEPS
        progress = (step - LEGACY_LR_WARMUP_STEPS) / max(1, total_steps - LEGACY_LR_WARMUP_STEPS)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return factor


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
    if args.precision == "bf16":
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # ---- dataset and policy ------------------------------------------------------
    dataset_dir = Path(args.dataset)
    repo_id = f"local/{dataset_dir.name}"
    ds_meta = LeRobotDatasetMetadata(repo_id, root=dataset_dir)
    if args.cameras:
        # The policy is built from these features: the other cameras are left out of its inputs
        # (the dataset and the env still provide them, unused).
        for name, key in CAMERA_FEATURES.items():
            if name not in args.cameras:
                ds_meta.info["features"].pop(key, None)
    policy_cfg = make_bc_policy_config(args, ds_meta)
    policy_cfg.device = device.type
    if args.normalizer == "per_dim_range":
        policy_cfg.normalization_mapping = {"VISUAL": NormalizationMode.MEAN_STD, "STATE": NormalizationMode.MIN_MAX,
                                            "ACTION": NormalizationMode.MIN_MAX}
        ds_meta.stats = per_dim_range_stats(ds_meta.stats, ds_meta.info["action_mode"])
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    image_transforms_cfg = ImageTransformsConfig(enable=True)
    # Frames and action chunks read from memory (see cached_dataset.py): same samples as
    # LeRobotDataset, without the per-sample video decoding that bottlenecked training.
    image_transforms = ImageTransforms(image_transforms_cfg)
    if args.random_shift > 0:
        h, w = ds_meta.features[ds_meta.camera_keys[0]]["shape"][:2]
        image_transforms = v2.Compose([image_transforms, v2.RandomCrop((h, w), padding=args.random_shift,
                                                                       padding_mode="edge")])
    dataset = CachedVideoLeRobotDataset(repo_id, root=dataset_dir, delta_timestamps=delta_timestamps,
                                        image_transforms=image_transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type != "cpu", drop_last=True, persistent_workers=args.num_workers > 0)
    dl_iter = cycle(dataloader)

    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    policy.train()
    if args.precision == "bf16":
        policy.to(memory_format=torch.channels_last)
    autocast = (lambda: torch.autocast(device.type, dtype=torch.bfloat16)) if args.precision == "bf16" \
        else contextlib.nullcontext
    if args.optimizer == "legacy":
        optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=policy_cfg.optimizer_lr, betas=(0.95, 0.999),
                                      eps=1e-8, weight_decay=policy_cfg.optimizer_weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, legacy_lr_lambda(args.steps))
    else:
        optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=policy_cfg.optimizer_lr,
                                      weight_decay=policy_cfg.optimizer_weight_decay)
        scheduler = None
    ema_policy = None
    if args.ema:
        ema_policy = copy.deepcopy(policy).eval()
        ema_policy.requires_grad_(False)
    eval_policy = ema_policy or policy

    step, best_success_rate, best_step, wandb_id = 0, -1.0, None, None
    if args.resume:
        state = load_latest(run_dir, policy, optimizer, device)
        step, best_success_rate, best_step, wandb_id = (state["step"], state["best_success_rate"],
                                                         state["best_step"], state["wandb_run_id"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        if ema_policy is not None:
            ema_policy.load_state_dict(state["ema"])
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
                "best_step": best_step, "wandb_run_id": wandb.run.id,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "ema": ema_policy.state_dict() if ema_policy is not None else None}

    # ---- training loop ----------------------------------------------------------------
    while step < args.steps:
        t0 = time.perf_counter()
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                 for k, v in next(dl_iter).items()}
        data_ms = (time.perf_counter() - t0) * 1000
        with autocast():
            loss, loss_dict = policy.forward(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip_norm or math.inf)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        step += 1
        if ema_policy is not None:
            with torch.no_grad():
                ema_params = [p for p in ema_policy.parameters()]
                torch._foreach_lerp_(ema_params, [p.detach() for p in policy.parameters()], 1.0 - ema_decay(step))

        if step % args.log_freq == 0:
            iter_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[step {step:>7d}/{args.steps}] loss {loss.item():.4f} | data {data_ms:.0f} ms | "
                        f"iter {iter_ms:.0f} ms")
            wandb.log({"train/loss": loss.item(), "train/grad_norm": float(grad_norm),
                       "train/lr": optimizer.param_groups[0]["lr"],
                       **{f"train/{k}": v for k, v in (loss_dict or {}).items()},
                       "time/data_ms": data_ms, "time/iter_ms": iter_ms}, step=step)

        if step % args.eval_freq == 0 or step == args.steps:
            video_path = run_dir / "evals" / f"step_{step:07d}.mp4"
            success_rate, mean_length, eval_s = evaluate(eval_policy, eval_env, args.eval_episodes, args.eval_seed,
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
                save_checkpoint(run_dir / "best", eval_policy,
                                {"eval.json": {"step": step, "success_rate": success_rate}})
                logger.info(colored(f"New best: saved {run_dir / 'best'}", "magenta"))

    eval_env.close()
    wandb.finish()
    logger.info(colored(f"Training finished. Best success rate {100 * best_success_rate:.1f} % at step "
                        f"{best_step}: {run_dir / 'best'}", "green", attrs=["bold"]))


if __name__ == "__main__":
    main()
