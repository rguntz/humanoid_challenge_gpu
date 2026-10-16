"""Convert the pick-and-place demos into a LeRobot dataset for ResFiT, and check it
by replaying every converted episode in the simulation env.

Input: videos/task/dataset_sim/policy/pick_place_bottle_base.hdf5, written by
resfit/pick_place/make_dataset.py: robomimic layout, observations and absolute actions
in the robot-base frame, plus the layout/ group the env samples its start states from.

Output, in --out-dir (default videos/task/dataset_sim/lerobot/): one LeRobot dataset
per action mode, <name>_<mode>/ (--name, default pick_place_bottle; modes absolute, delta,
absolute_6d), identical
except for their actions, with one episode per demo at the demos' control rate (15 Hz):
  observation.state              (9)   eef position (3), eef quaternion x y z w (4),
                                       finger joint positions (2)
  observation.images.agentview   (128, 128, 3) video, the fixed camera across the table
  observation.images.wrist       (128, 128, 3) video, the camera on the robot's wrist
  action                         (7)   absolute or delta end-effector action, or
                                 (10)  absolute_6d: absolute, 6D rotation, or
                                 (8)   joint: the demo's recorded joint targets (7) and
                                       gripper, with observation.state the arm joints (7)
                                       and fingers (2) instead of the end-effector pose
                                       (see resfit/pick_place/action_mode.py)
  next.done                      (1)   true on the last frame of each episode
  meta/info.json                 "action_mode": the mode the actions are stored in,
                                 "state_mode": eef or joint,
                                 "task": the language instruction, read from here by
                                         training and evaluation alike (--task to change it)
  meta/source_demos.json         per episode: the source demo and its row in the
                                 layout/ group (the env's reset(options={"demo": row}))

The state has no object pose: like ResFiT's camera-based tasks, the policy only gets
the gripper's own state and the camera images (and ACT would use any
observation.environment_state feature as an input).

Each episode ends on the demo's first success (bottle released in the container), as
an env episode does: ResFiT rewards the last frame of each offline episode
(reward = done), and the env rewards and ends an episode on its first success.
--full-episodes keeps every frame of the demos instead, the frames after the first success
included, as policy/ trains on them (behaviour cloning only: the last frame is then no
longer the success, so not for ResFiT's offline rewards); meta/info.json then has
"full_episodes": true.

Checks, on the dataset as written (read back with LeRobotDataset):
  1. The actions turned back into the env's absolute targets (action_mode.to_env,
     from the dataset's own observation.state) equal the source's absolute actions;
     joint actions equal the source's recorded joint targets.
  2. Every episode replayed in the env (resfit/pick_place/env.py), from the demo's
     layout, reaches success. Delta actions are turned into absolute targets from the
     recorded pose (observation.state), so the replay follows the demo; absolute_6d
     actions go through the 6D -> rotation conversion the env adapter uses; joint
     actions go to the env's step_joint (no IK), as the recording executed them. After the
     episode's last action the target is held for up to HOLD_STEPS steps: the env's
     success can come a step after the recording's.
  3. The env's first observation equals the episode's first observation.state, the
     env's renders of both cameras at reset match the episode's first images (so the
     policy sees in the env what it was trained on), and the first and last video
     frames of every episode match the source images.
  Reported only, in delta mode: the same replay with delta actions applied from the
  env's live pose, as the env adapter does with a policy. Without a policy to correct
  them, the small differences between the env (end-effector targets through IK) and
  the recording (joint targets) add up step after step, so this replay drifts; the
  numbers say how much a delta policy has to correct.

The check's replays, both cameras side by side, are saved in each dataset's
folder, in check_videos/: episode_<i>_replay.mp4 (check 2) and, for the delta
dataset, episode_<i>_live.mp4 (the replay from the live pose).

resfit/pick_place/make_dataset.py runs this conversion after writing its dataset; run
this script alone to redo only the LeRobot datasets (all, or the given modes):
    uv run resfit/lerobot/dataset/convert_pick_place_to_lerobot.py
    uv run resfit/lerobot/dataset/convert_pick_place_to_lerobot.py --modes absolute_6d
    uv run resfit/lerobot/dataset/convert_pick_place_to_lerobot.py --full-episodes \
        --name pick_place_bottle_full --modes absolute_6d
The regenerated versions (resfit/pick_place/regenerate_dataset.py, which runs this conversion
itself), e.g. the one that starts from a fixed home pose:
    uv run resfit/lerobot/dataset/convert_pick_place_to_lerobot.py \
        --src videos/task/dataset_sim/policy/pick_place_bottle_home.hdf5 --name pick_place_bottle_home
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering of the replays, before mujoco is imported

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.common.datasets.utils import write_info  # noqa: E402

from resfit.pick_place.action_mode import ACTION_DIMS, ACTION_MODES, STATE_MODES, from_absolute, to_env  # noqa: E402
from resfit.pick_place.task import IMAGE_FEATURES as IMAGE_KEYS  # noqa: E402

SIM = REPO / "videos" / "task" / "dataset_sim"
STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
STATE_NAMES = {
    "eef": ["eef_pos_x", "eef_pos_y", "eef_pos_z", "eef_quat_x", "eef_quat_y", "eef_quat_z", "eef_quat_w",
            "finger_qpos_0", "finger_qpos_1"],
    "joint": [f"joint_pos_{i}" for i in range(7)] + ["finger_qpos_0", "finger_qpos_1"],
}
ARM_JOINTS = 7
# The recorded demos (sim/rollout_physics.py's Recorder): their joint targets, actions_joint,
# for the datasets converted from make_dataset.py's output, which does not copy them.
RECORDED = SIM / "robomimic" / "pick_place_bottle.hdf5"
ACTION_NAMES = {
    "absolute": ["eef_pos_x", "eef_pos_y", "eef_pos_z", "eef_rot_x", "eef_rot_y", "eef_rot_z", "gripper"],
    "delta": ["eef_delta_pos_x", "eef_delta_pos_y", "eef_delta_pos_z",
              "eef_delta_rot_x", "eef_delta_rot_y", "eef_delta_rot_z", "gripper"],
    "absolute_6d": ["eef_pos_x", "eef_pos_y", "eef_pos_z", "eef_rot6d_r0x", "eef_rot6d_r0y", "eef_rot6d_r0z",
                    "eef_rot6d_r1x", "eef_rot6d_r1y", "eef_rot6d_r1z", "gripper"],
    "joint": [f"joint_target_{i}" for i in range(7)] + ["gripper"],
}


def joint_actions(name, d, T):
    """The demo's recorded joint targets (T, 8): from the source itself when it has them
    (regenerate_dataset.py's output), else from the recording, checked to be the same demo
    (same simulator states)."""
    if "actions_joint" in d:
        return d["actions_joint"][:T].astype(np.float32)
    with h5py.File(RECORDED, "r") as rec:
        r = rec["data"][name]
        # The robot's joints and fingers (qpos[:9]) do not depend on the scene's frame.
        if not np.allclose(r["states"][:T, 1:1 + ARM_JOINTS + 2], d["states"][:T, 1:1 + ARM_JOINTS + 2], atol=1e-9):
            raise SystemExit(f"{name}: the recording's states differ from the dataset's")
        return r["actions_joint"][:T].astype(np.float32)


def episode_state(d, T, mode):
    """observation.state (T, 9): end-effector pose and fingers, or arm and finger joints."""
    if STATE_MODES[mode] == "joint":
        return d["states"][:T, 1:1 + ARM_JOINTS + 2].astype(np.float32)  # rows: [time, qpos, qvel]
    return np.concatenate([d["obs"][k][:T] for k in STATE_KEYS], axis=1).astype(np.float32)
TASK = "pick up the pill box and place it in the container"
STEPS_AFTER_SUCCESS = 0  # frames kept after each demo's first success
HOLD_STEPS = 8  # steps the replay check holds the last target after an episode's last action
RENDER_TOLERANCE = 0.02  # env render vs recorded image at reset, mean abs error (pixels in [0, 1])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(SIM / "policy" / "pick_place_bottle_base.hdf5"))
    p.add_argument("--out-dir", default=str(SIM / "lerobot"), help="folder the datasets are written in")
    p.add_argument("--name", default="pick_place_bottle", help="datasets are written as <name>_<mode>/")
    p.add_argument("--modes", nargs="+", default=list(ACTION_MODES), choices=ACTION_MODES,
                   help="action modes to write (default: all)")
    p.add_argument("--full-episodes", action="store_true",
                   help="keep the frames after each demo's first success (see the module docstring)")
    p.add_argument("--task", default=TASK,
                   help="language instruction encoded in every episode (meta/info.json's 'task'); "
                        "training and evaluation both read it from there, so it only needs setting here")
    return p.parse_args()


def demo_names(f):
    return sorted(f["data"], key=lambda n: int(n.split("_")[1]))


def convert(src, mode, out, repo_id, full_episodes=False, task=TASK):
    with h5py.File(src, "r") as f:
        names = demo_names(f)
        layout_rows = {n: i for i, n in enumerate(json.loads(f["layout"].attrs["demos"]))}
        first = f["data"][names[0]]
        fps = float(first.attrs["control_freq"])
        if fps != int(fps):
            raise SystemExit(f"LeRobot needs an integer fps, the demos run at {fps} Hz")
        features = {
            "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES[STATE_MODES[mode]]),),
                                  "names": STATE_NAMES[STATE_MODES[mode]]},
            **{key: {"dtype": "video", "shape": first["obs"][camera].shape[1:], "names": ["height", "width", "channel"]}
               for camera, key in IMAGE_KEYS.items()},
            "action": {"dtype": "float32", "shape": (ACTION_DIMS[mode],), "names": ACTION_NAMES[mode]},
            "next.done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
        }
        ds = LeRobotDataset.create(repo_id=repo_id, fps=int(fps), root=out, robot_type="panda",
                                   features=features, use_videos=True, image_writer_threads=4)

        sources, kept, total = [], 0, 0
        for name in names:
            d = f["data"][name]
            rewards = d["rewards"][:]
            if not (rewards > 0).any():
                raise SystemExit(f"{name} never succeeds")
            T = len(rewards) if full_episodes else \
                min(len(rewards), int(np.argmax(rewards > 0)) + 1 + STEPS_AFTER_SUCCESS)
            state = episode_state(d, T, mode)
            if mode == "joint":
                actions = joint_actions(name, d, T)
            else:
                actions = from_absolute(d["actions"][:T].astype(np.float32), mode, state[:, 0:3], state[:, 3:7])
            images = {key: d["obs"][camera][:T] for camera, key in IMAGE_KEYS.items()}
            for t in range(T):
                ds.add_frame({"observation.state": state[t], **{k: v[t] for k, v in images.items()},
                              "action": actions[t], "next.done": np.array([t == T - 1])}, task=task)
            ds.save_episode()
            sources.append({"episode_index": len(sources), "demo": name, "layout_row": layout_rows[name],
                            "length": T, "first_success": int(np.argmax(rewards > 0))})
            kept, total = kept + T, total + len(rewards)
            print(f"  {name}: {T} frames", flush=True)

    ds.meta.info["action_mode"] = mode
    ds.meta.info["state_mode"] = STATE_MODES[mode]
    ds.meta.info["full_episodes"] = full_episodes
    ds.meta.info["task"] = task
    ds.meta.info["source"] = str(Path(src).resolve().relative_to(REPO))
    write_info(ds.meta.info, ds.root)
    (Path(out) / "meta" / "source_demos.json").write_text(json.dumps(sources, indent=1))
    print(f"{len(names)} episodes, {kept} frames ({kept}/{total} of the source's), "
          f"{mode} actions -> {out}")


def replay(env, layout_row, actions, states, mode, reference, hold_steps, frames=None):
    """Replay one episode's actions in the env (see the module docstring). Returns the
    step of the first success (None if none), the env's first observation.state, and the
    largest end-effector position (m) and rotation (deg) gaps to the recorded states
    (end-effector states only). Appends the rendered frames to `frames` if given."""
    joint = STATE_MODES[mode] == "joint"

    def observed(obs):  # the env's observation.state, as the dataset stores it
        return env.data.qpos[:ARM_JOINTS + 2].copy() if joint else obs[:states.shape[1]].copy()

    obs, _ = env.reset(options={"demo": layout_row})
    if frames is not None:
        frames.append(env.render())
    first_obs, n = observed(obs), len(actions)
    pos_gap, rot_gap = 0.0, 0.0
    for t in range(n + hold_steps):
        k = min(t, n - 1)
        if joint:
            obs, _, _, truncated, info = env.step_joint(actions[k])
        else:
            live = observed(obs)
            pos_gap = max(pos_gap, float(np.linalg.norm(live[:3] - states[k, :3])))
            rot_gap = max(rot_gap, float(np.degrees(
                (Rotation.from_quat(live[3:7]) * Rotation.from_quat(states[k, 3:7]).inv()).magnitude())))
            ref = live if reference == "live" else states[k]
            obs, _, _, truncated, info = env.step(to_env(actions[k], mode, ref[0:3], ref[3:7]))
        if frames is not None:
            frames.append(env.render())
        if info["success"]:
            return t, first_obs, pos_gap, rot_gap
        if truncated:
            break
    return None, first_obs, pos_gap, rot_gap


def check(src, out, repo_id):
    """Check the written dataset (see the module docstring)."""
    from resfit.pick_place.env import PickPlaceBottleEnv
    from resfit.pick_place.make_dataset import save_video

    ds = LeRobotDataset(repo_id, root=out)
    mode = ds.meta.info["action_mode"]
    episodes = json.loads((Path(out) / "meta" / "source_demos.json").read_text())
    actions = torch.stack(ds.hf_dataset["action"]).numpy()
    states = torch.stack(ds.hf_dataset["observation.state"]).numpy()
    starts, ends = ds.episode_data_index["from"].tolist(), ds.episode_data_index["to"].tolist()
    assert len(episodes) == ds.meta.total_episodes == len(starts)

    # Without cameras the env's observation is the full low-dim state; its first 9
    # entries are observation.state (task.py's flat_obs order).
    env = PickPlaceBottleEnv(src, render=True)
    video_dir = Path(out) / "check_videos"
    video_dir.mkdir()
    fps = ds.meta.fps
    failed, action_err, first_obs_err, image_err, render_err = [], 0.0, 0.0, 0.0, 0.0
    live_ok, live_pos_gap, live_rot_gap = 0, 0.0, 0.0
    with h5py.File(src, "r") as f:
        for ep, i0, i1 in zip(episodes, starts, ends):
            a, s = actions[i0:i1], states[i0:i1]
            source = f["data"][ep["demo"]]
            if mode == "joint":
                action_err = max(action_err, float(np.abs(a - joint_actions(ep["demo"], source, len(a))).max()))
            else:
                a_abs = to_env(a, mode, s[:, 0:3], s[:, 3:7])
                # Compare rotations, not axis-angle vectors (several vectors give one rotation).
                src_abs = source["actions"][:len(a)]
                rot_err = (Rotation.from_rotvec(a_abs[:, 3:6]) * Rotation.from_rotvec(src_abs[:, 3:6]).inv()).magnitude()
                action_err = max(action_err, float(np.abs(a_abs[:, :3] - src_abs[:, :3]).max()),
                                 float(rot_err.max()), float(np.abs(a_abs[:, 6] - src_abs[:, 6]).max()))

            frames = []
            step, first_obs, _, _ = replay(env, ep["layout_row"], a, s, mode, "recorded", HOLD_STEPS, frames)
            save_video(frames, video_dir / f"episode_{ep['episode_index']:03d}_replay.mp4", fps)
            first_obs_err = max(first_obs_err, float(np.abs(first_obs - s[0]).max()))
            if step is None:
                failed.append(ep["demo"])
            line = (f"  episode {ep['episode_index']} ({ep['demo']}): "
                    f"{'success at step ' + str(step) if step is not None else 'FAILED'} "
                    f"(demo's first success: step {ep['first_success']})")
            if mode == "delta":
                frames = []
                live_step, _, pos_gap, rot_gap = replay(env, ep["layout_row"], a, s, mode, "live", HOLD_STEPS, frames)
                save_video(frames, video_dir / f"episode_{ep['episode_index']:03d}_live.mp4", fps)
                live_ok += live_step is not None
                live_pos_gap, live_rot_gap = max(live_pos_gap, pos_gap), max(live_rot_gap, rot_gap)
                line += (f"; from the live pose: {'success' if live_step is not None else 'no success'}, "
                         f"drift up to {100 * pos_gap:.1f} cm {rot_gap:.0f} deg")
            print(line, flush=True)

            env.reset(options={"demo": ep["layout_row"]})
            for camera, rendered in env.get_images(IMAGE_KEYS).items():
                render_err = max(render_err, float(np.abs(rendered / 255.0 - source["obs"][camera][0] / 255.0).mean()))
            for t in (0, i1 - i0 - 1):
                frame = ds[i0 + t]
                for camera, key in IMAGE_KEYS.items():
                    decoded = frame[key].permute(1, 2, 0).numpy()  # (H, W, C) in [0, 1]
                    image_err = max(image_err, float(np.abs(decoded - source["obs"][camera][t] / 255.0).mean()))
    env.close()

    n = len(episodes)
    print(f"actions vs the source's ({'joint targets' if mode == 'joint' else 'absolute actions'}), "
          f"max error: {action_err:.1e}")
    print(f"replayed {n} episodes ({mode} actions): {n - len(failed)} succeed")
    print(f"env's first observation vs observation.state, max abs error: {first_obs_err:.1e}")
    print(f"env's renders at reset vs the episodes' first images, worst mean abs error: {render_err:.4f}")
    print(f"decoded video vs source images, worst mean abs error per frame: {image_err:.4f} (video compression)")
    print(f"replay videos -> {video_dir}")
    if mode == "delta":
        print(f"reported only: delta actions applied from the env's live pose: {live_ok}/{n} succeed, "
              f"drift from the recorded states up to {100 * live_pos_gap:.1f} cm, {live_rot_gap:.0f} deg")
    if action_err > 1e-5:
        raise SystemExit("The dataset's actions do not give back the source's absolute actions.")
    if failed:
        raise SystemExit(f"These episodes do not reach success when replayed: {failed}")
    if first_obs_err > 1e-4:
        raise SystemExit("The env's first observation does not match the dataset's.")
    if render_err > RENDER_TOLERANCE:
        raise SystemExit("The env does not render the cameras as the demos were recorded.")
    if image_err > 0.05:
        raise SystemExit("The decoded video frames do not match the source images.")


def convert_all(src, out_dir, modes=ACTION_MODES, name_prefix="pick_place_bottle", full_episodes=False, task=TASK):
    """Write and check one dataset per action mode (see the module docstring)."""
    for mode in modes:
        name = f"{name_prefix}_{mode}"
        out, repo_id = Path(out_dir) / name, f"local/{name}"
        print(f"=== {mode} actions -> {out}")
        if out.exists():
            shutil.rmtree(out)
        convert(src, mode, out, repo_id, full_episodes, task)
        check(src, out, repo_id)


def main():
    args = parse_args()
    convert_all(args.src, args.out_dir, args.modes, args.name, args.full_episodes, args.task)


if __name__ == "__main__":
    main()
