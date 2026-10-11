"""Check that the observations the policy gets in the simulator reproduce the ones it
was trained on.

Every episode of the LeRobot dataset is replayed in the simulator through
PickPlaceGymEnv (vec_env.py), the same env class and observation code the policy is
evaluated with: reset to the demo's layout, then the episode's recorded actions one
by one. Before each action, the simulator's observation is compared with the
dataset's frame at the same step, as LeRobotDataset gives it to training (decoded
from the dataset's videos, no augmentation):
  images   observation.images.agentview and observation.images.wrist, (3, H, W) in
           [0, 1]: mean squared error and PSNR per camera
  state    observation.state (eef position, quaternion, finger joints): mean squared
           error, plus the end-effector position error (mm), rotation error (deg) and
           finger error (mm)
As a reference for the image numbers, the dataset frames are also compared with the
source images they were encoded from (pick_place_bottle_base.hdf5): that part of the
error is video compression, which the simulator's frames do not have.

What to expect: at step 0 (right after reset) the simulator is in the recorded state,
so its observation should match the dataset's up to compression. Later steps drift a
little, because the simulator reaches each target through IK while the demos were
recorded with joint targets (millimetres, see convert_pick_place_to_lerobot.py); the
images then differ where the arm and bottle moved.

Use the absolute dataset (the default): its actions replay the demo open-loop, so the
simulator follows the recorded trajectory. Delta actions are applied from the
simulator's own pose and drift by centimetres without a policy to correct them.

Outputs, in outputs/rollout_obs_check/<dataset name>/:
  summary.json        the numbers printed at the end
  per_episode.csv     per episode: step-0 and mean values of every metric
  per_step.png        every metric against the step, mean over the episodes
  videos/episode_<i>.mp4   per step, for each camera (rows): simulator | dataset |
                      absolute difference x4

Needs MuJoCo rendering: a GPU (MUJOCO_GL=egl, the default here) or, slower, Mesa's
software renderer. Run from the repository root:
    uv run resfit/pick_place/check_rollout_obs.py
    uv run resfit/pick_place/check_rollout_obs.py --dataset videos/task/dataset_sim/lerobot/pick_place_bottle_delta
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # before mujoco is imported

import h5py
import imageio
import matplotlib
import numpy as np
import torch
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from resfit.lerobot.dataset.cached_dataset import CachedVideoLeRobotDataset  # noqa: E402
from resfit.pick_place.env import CAMERA_KEYS  # noqa: E402
from resfit.pick_place.task import IMAGE_FEATURES  # noqa: E402
from resfit.pick_place.vec_env import PickPlaceGymEnv, dataset_env_config  # noqa: E402

LEROBOT = REPO / "videos" / "task" / "dataset_sim" / "lerobot"
CAMERAS = {IMAGE_FEATURES[k]: k for k in CAMERA_KEYS}  # LeRobot feature -> source image key
DIFF_GAIN = 4  # the difference panel of the videos is |sim - dataset| x DIFF_GAIN


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=str(LEROBOT / "pick_place_bottle_absolute"), help="LeRobot dataset folder")
    p.add_argument("--out", default=None, help="output folder (default outputs/rollout_obs_check/<dataset name>)")
    return p.parse_args()


def psnr(mse):
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def state_errors(sim, ref):
    """observation.state errors: MSE over the 9 values, eef position (mm), rotation (deg), fingers (mm)."""
    rot = (Rotation.from_quat(sim[3:7]) * Rotation.from_quat(ref[3:7]).inv()).magnitude()
    return {
        "state_mse": float(np.mean((sim - ref) ** 2)),
        "eef_pos_mm": float(1000 * np.linalg.norm(sim[:3] - ref[:3])),
        "eef_rot_deg": float(np.degrees(rot)),
        "finger_mm": float(1000 * np.abs(sim[7:9] - ref[7:9]).max()),
    }


def video_frame(sim_obs, ref):
    """One video frame: a row per camera, simulator | dataset | difference x DIFF_GAIN."""
    rows = []
    for key in CAMERAS:
        s = sim_obs[key].transpose(1, 2, 0)
        r = ref[key].transpose(1, 2, 0)
        d = np.clip(np.abs(s - r) * DIFF_GAIN, 0, 1)
        rows.append(np.hstack([s, r, d]))
    return (np.vstack(rows) * 255).round().astype(np.uint8)


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset).resolve()
    out = Path(args.out) if args.out else REPO / "outputs" / "rollout_obs_check" / dataset_dir.name
    (out / "videos").mkdir(parents=True, exist_ok=True)

    action_mode, source = dataset_env_config(dataset_dir)
    ds = CachedVideoLeRobotDataset(f"local/{dataset_dir.name}", root=dataset_dir)  # no augmentation
    episodes = json.loads((dataset_dir / "meta" / "source_demos.json").read_text())
    actions = torch.stack(ds.hf_dataset["action"]).numpy()
    starts, ends = ds.episode_data_index["from"].tolist(), ds.episode_data_index["to"].tolist()
    env = PickPlaceGymEnv(source, action_mode)
    print(f"{dataset_dir.name}: {len(episodes)} episodes, {action_mode} actions, rendering with "
          f"MUJOCO_GL={os.environ['MUJOCO_GL']} -> {out}", flush=True)

    metric_names = ([f"{cam}_{m}" for cam in CAMERAS for m in ("mse", "psnr")]
                    + ["state_mse", "eef_pos_mm", "eef_rot_deg", "finger_mm"]
                    + [f"{cam}_compression_mse" for cam in CAMERAS])
    per_step = {}  # step -> metric -> list over episodes
    rows = []
    with h5py.File(source, "r") as f:
        for ep, i0, i1 in zip(episodes, starts, ends):
            source_images = {cam: f["data"][ep["demo"]]["obs"][k] for cam, k in CAMERAS.items()}
            sim_obs, _ = env.reset(options={"demo": ep["layout_row"]})
            writer = imageio.get_writer((out / "videos" / f"episode_{ep['episode_index']:03d}.mp4").as_posix(),
                                        fps=env.metadata["render_fps"])
            episode_metrics = []
            for t in range(i1 - i0):
                frame = ds[i0 + t]
                ref = {k: frame[k].numpy() for k in [*CAMERAS, "observation.state"]}
                m = {}
                for cam, key in CAMERAS.items():
                    m[f"{cam}_mse"] = float(np.mean((sim_obs[cam] - ref[cam]) ** 2))
                    m[f"{cam}_psnr"] = psnr(m[f"{cam}_mse"])
                    raw = source_images[cam][t].transpose(2, 0, 1) / 255.0
                    m[f"{cam}_compression_mse"] = float(np.mean((ref[cam] - raw) ** 2))
                m.update(state_errors(sim_obs["observation.state"], ref["observation.state"]))
                episode_metrics.append(m)
                for name, value in m.items():
                    per_step.setdefault(t, {}).setdefault(name, []).append(value)
                writer.append_data(video_frame(sim_obs, ref))
                if t < i1 - i0 - 1:
                    sim_obs, _, _, _, _ = env.step(actions[i0 + t])
            writer.close()

            row = {"episode": ep["episode_index"], "demo": ep["demo"], "steps": i1 - i0}
            for name in metric_names:
                values = [m[name] for m in episode_metrics]
                row[f"{name}_step0"] = values[0]
                row[f"{name}_mean"] = float(np.mean(values))
            rows.append(row)
            print(f"  episode {ep['episode_index']:2d}: step 0 / mean "
                  + " | ".join(f"{cam.split('.')[-1]} MSE {row[f'{cam}_mse_step0']:.1e} / {row[f'{cam}_mse_mean']:.1e}"
                               for cam in CAMERAS)
                  + f" | eef {row['eef_pos_mm_step0']:.2f} / {row['eef_pos_mm_mean']:.2f} mm", flush=True)
    env.close()

    with open(out / "per_episode.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"dataset": str(dataset_dir), "action_mode": action_mode, "episodes": len(rows)}
    for name in metric_names:
        step0 = [r[f"{name}_step0"] for r in rows]
        mean = [r[f"{name}_mean"] for r in rows]
        summary[name] = {"step0_mean": float(np.mean(step0)), "step0_max": float(np.max(step0)),
                         "all_steps_mean": float(np.mean(mean)), "worst_episode_mean": float(np.max(mean))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    steps = sorted(per_step)
    panels = [("image MSE (sim vs dataset; dashed: compression)", [f"{c}_mse" for c in CAMERAS]),
              ("eef position error (mm)", ["eef_pos_mm"]), ("eef rotation error (deg)", ["eef_rot_deg"]),
              ("finger error (mm)", ["finger_mm"])]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 3 * len(panels)), sharex=True)
    for ax, (title, names) in zip(axes, panels):
        for name in names:
            ax.plot(steps, [np.mean(per_step[t][name]) for t in steps], label=name)
        if names[0].endswith("_mse") and names[0] != "state_mse":
            for cam in CAMERAS:
                ax.plot(steps, [np.mean(per_step[t][f"{cam}_compression_mse"]) for t in steps], "--",
                        label=f"{cam}_compression_mse")
        ax.set_title(title)
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("step (mean over the episodes still running)")
    fig.tight_layout()
    fig.savefig(out / "per_step.png", dpi=120)

    print(f"\n{len(rows)} episodes ({action_mode} actions). Mean over episodes: step 0 | all steps | worst episode")
    for name in metric_names:
        s = summary[name]
        print(f"  {name:45s} {s['step0_mean']:10.3e} | {s['all_steps_mean']:10.3e} | {s['worst_episode_mean']:10.3e}")
    print(f"Saved {out}/: summary.json, per_episode.csv, per_step.png, videos/")


if __name__ == "__main__":
    main()
