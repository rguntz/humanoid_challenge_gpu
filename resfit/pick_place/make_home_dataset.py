"""Write the pick-and-place dataset in a version where every episode starts from the same
home pose, then its LeRobot datasets.

In the demos (pick_place_bottle_base.hdf5, from make_dataset.py) the arm starts wherever
the person's hand was when the video started: the end-effector start varies by up to
25 cm and 167 deg between demos. A robot starts from a fixed pose instead. Here each demo
is regenerated in the simulator (PickPlaceBottleEnv, the evaluation env) from that home
pose:
  1. reset to the demo's layout (bottle, container, table), with the arm at the home pose
  2. approach: the arm moves from the home pose to the demo's start configuration. The
     path is interpolated in joint space (smoothstep, zero velocity at both ends, so it
     ends on the demo's own arm configuration, with no wrist flip or IK branch change)
     and sent as the end-effector targets of those joint positions (forward kinematics),
     like any action. Its length follows the distance: at most APPROACH_SPEED m/s,
     APPROACH_ROT_SPEED deg/s and APPROACH_JOINT_SPEED rad/s on average, at least
     APPROACH_MIN_STEPS steps; the gripper stays open.
  3. the demo's own absolute actions, then, if it has not succeeded yet, its last action
     held for up to HOLD_STEPS steps. The episode is cut steps_after_success steps after
     the first success, as the source dataset is.
Everything the policy gets (observations, both cameras, rewards) is recorded from the
simulator, so the training images are rendered exactly as the evaluation's. A demo that
does not reach success is left out (reported).

The home pose is the medoid of the demos' start configurations: the demo start with the
smallest summed joint-space distance to all the others (a real start, unlike a
joint-wise mean). Output, next to the source (they share env_scene.xml):
  pick_place_bottle_home.hdf5  same layout as the source; layout/robot_qpos is the home
      pose for every demo, so the env starts every episode (evaluation included) there.
      layout.attrs: home_qpos, home_from_demo. Per demo, attrs approach_steps and
      approach_end_error_mm / _deg (end-effector error to the demo's start after the
      approach); states are [time, qpos, qvel] of env_scene.xml (the source's states are
      of each demo's own scene).
Then the LeRobot datasets pick_place_bottle_home_<mode>/ (convert_pick_place_to_lerobot.py,
with its replay checks and check videos).

Usage, after make_dataset.py:
    uv run resfit/pick_place/make_home_dataset.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # before mujoco is imported

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from resfit.pick_place import task  # noqa: E402
from resfit.pick_place.env import CAMERA_KEYS, PickPlaceBottleEnv  # noqa: E402

POLICY_DIR = REPO / "videos" / "task" / "dataset_sim" / "policy"
APPROACH_SPEED = 0.15  # m/s, mean end-effector speed of the approach
APPROACH_ROT_SPEED = 90.0  # deg/s, mean end-effector rotation speed
APPROACH_JOINT_SPEED = 1.0  # rad/s, mean speed of the fastest joint
APPROACH_MIN_STEPS = 8
HOLD_STEPS = 8  # steps the last demo action is held for if the demo has not succeeded yet


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(POLICY_DIR / "pick_place_bottle_base.hdf5"))
    p.add_argument("--out", default=str(POLICY_DIR / "pick_place_bottle_home.hdf5"),
                   help="must be in the source's folder (env_scene.xml)")
    p.add_argument("--lerobot-dir", default=str(REPO / "videos" / "task" / "dataset_sim" / "lerobot"))
    p.add_argument("--name", default="pick_place_bottle_home", help="LeRobot datasets: <name>_<mode>/")
    p.add_argument("--no-lerobot", action="store_true", help="only write the hdf5")
    return p.parse_args()


def medoid(q):
    """Index of the row with the smallest summed distance to all the others."""
    d = np.linalg.norm(q[:, None] - q[None], axis=-1)
    return int(np.argmin(d.sum(1)))


def eef_base_pose(env, fk_data, q_arm):
    """End-effector (tool centre point) position and rotation in the base frame for the
    arm joint positions q_arm (forward kinematics)."""
    fk_data.qpos[:task.ARM] = q_arm
    mujoco.mj_kinematics(env.model, fk_data)
    pos = task.to_base_pos(fk_data.site_xpos[env.scene.site], env.pb, env.Rb)
    rot = Rotation.from_matrix(env.Rb.T @ fk_data.site_xmat[env.scene.site].reshape(3, 3))
    return pos, rot


def approach_actions(env, fk_data, q_from, q_to, gripper, fps):
    """Absolute base-frame actions (n, 7) moving the arm from q_from to q_to (see the module
    docstring)."""
    p0, r0 = eef_base_pose(env, fk_data, q_from)
    p1, r1 = eef_base_pose(env, fk_data, q_to)
    seconds = max(np.linalg.norm(p1 - p0) / APPROACH_SPEED,
                  np.degrees((r1 * r0.inv()).magnitude()) / APPROACH_ROT_SPEED,
                  np.abs(q_to - q_from).max() / APPROACH_JOINT_SPEED)
    n = max(APPROACH_MIN_STEPS, int(np.ceil(seconds * fps)))
    actions = []
    for k in range(1, n + 1):
        u = k / n
        s = 3 * u ** 2 - 2 * u ** 3
        pos, rot = eef_base_pose(env, fk_data, q_from + s * (q_to - q_from))
        actions.append(np.concatenate([pos, rot.as_rotvec(), [gripper]]))
    return np.array(actions, np.float32)


def record(env):
    obs = task.obs_to_base(task.world_lowdim_obs(env.data, env.scene), env.pb, env.Rb)
    images = env.get_images(CAMERA_KEYS)
    state = np.concatenate([[env.data.time], env.data.qpos, env.data.qvel])
    return obs, images, state


def rollout(env, fk_data, row, home, demo_actions, steps_after_success, fps):
    """One demo from the home pose (see the module docstring). Returns the episode's
    arrays (None if it never succeeds) and the approach's length and end error."""
    env.reset(options={"demo": row})
    q_start = env.layouts["robot_qpos_demo"][row][:task.ARM]
    approach = approach_actions(env, fk_data, home[:task.ARM], q_start, demo_actions[0, 6], fps)
    plan = list(approach) + list(demo_actions) + [demo_actions[-1]] * HOLD_STEPS

    frames, first_success, approach_err = [], None, None
    for t, action in enumerate(plan):
        if t == len(approach):
            p, r = eef_base_pose(env, fk_data, q_start)
            live = task.obs_to_base(task.world_lowdim_obs(env.data, env.scene), env.pb, env.Rb)
            approach_err = (1000 * float(np.linalg.norm(live["robot0_eef_pos"] - p)),
                            float(np.degrees((Rotation.from_quat(live["robot0_eef_quat"]) * r.inv()).magnitude())))
        obs, images, state = record(env)
        _, reward, success, _, _ = env.step(action)
        frames.append((obs, images, state, action, reward))
        if success and first_success is None:
            first_success = t
        if first_success is not None and t >= first_success + steps_after_success:
            break
    if first_success is None:
        return None, len(approach), approach_err
    T = len(frames)
    episode = {
        "obs": {k: np.stack([f[0][k] for f in frames]) for k in task.LOWDIM_KEYS},
        "images": {k: np.stack([f[1][k] for f in frames]) for k in CAMERA_KEYS},
        "states": np.stack([f[2] for f in frames]),
        "actions": np.stack([f[3] for f in frames]).astype(np.float32),
        "rewards": np.array([f[4] for f in frames], np.float32),
        "dones": np.eye(T, dtype=np.int64)[-1],
        "first_success": first_success,
    }
    return episode, len(approach), approach_err


def main():
    args = parse_args()
    src, out = Path(args.src), Path(args.out)
    if out.parent.resolve() != src.parent.resolve():
        raise SystemExit("--out must be in the source's folder: the env loads env_scene.xml from there")

    with h5py.File(src, "r") as f:
        layout_demos = json.loads(f["layout"].attrs["demos"])
        layouts = {k: f["layout"][k][:] for k in f["layout"]}
        steps_after_success = int(f["data"].attrs["steps_after_success"])
        fps = float(f["data"][layout_demos[0]].attrs["control_freq"])
    q_starts = layouts["robot_qpos"]
    home_row = medoid(q_starts[:, :task.ARM])
    home = q_starts[home_row].copy()

    env = PickPlaceBottleEnv(src, camera_names=list(CAMERA_KEYS))
    env.layouts["robot_qpos_demo"] = env.layouts["robot_qpos"].copy()
    env.layouts["robot_qpos"][:] = home  # the env's resets start from the home pose
    fk_data = mujoco.MjData(env.model)
    p_home, r_home = eef_base_pose(env, fk_data, home[:task.ARM])
    print(f"home pose: start of {layout_demos[home_row]} (medoid of the {len(q_starts)} demo starts), "
          f"end-effector at {np.round(p_home, 3)} m (base frame)", flush=True)

    kept, failed, approach_steps, approach_errs = [], [], [], []
    with h5py.File(src, "r") as fs, h5py.File(out, "w") as fd:
        data_out = fd.create_group("data")
        for k, v in fs["data"].attrs.items():
            data_out.attrs[k] = v
        total = 0
        for row, name in enumerate(layout_demos):
            d = fs["data"][name]
            episode, n_approach, err = rollout(env, fk_data, row, home, d["actions"][:], steps_after_success, fps)
            approach_steps.append(n_approach)
            approach_errs.append(err)
            if episode is None:
                failed.append(name)
                print(f"  {name}: approach {n_approach} steps (end error {err[0]:.1f} mm, {err[1]:.1f} deg), "
                      "NO SUCCESS: left out", flush=True)
                continue
            g = data_out.create_group(name)
            for k, v in d.attrs.items():
                g.attrs[k] = v
            T = len(episode["actions"])
            g.attrs["num_samples"] = T
            g.attrs["num_samples_recorded"] = T
            g.attrs["approach_steps"] = n_approach
            g.attrs["approach_end_error_mm"], g.attrs["approach_end_error_deg"] = err
            for k, v in episode["obs"].items():
                g.create_dataset(f"obs/{k}", data=v)
            for k, v in episode["images"].items():
                g.create_dataset(f"obs/{k}", data=v, compression="gzip", compression_opts=4,
                                 chunks=(1,) + v.shape[1:])
            for k in ("actions", "rewards", "dones", "states"):
                g.create_dataset(k, data=episode[k])
            kept.append(row)
            total += T
            print(f"  {name}: approach {n_approach} steps (end error {err[0]:.1f} mm, {err[1]:.1f} deg), "
                  f"success at step {episode['first_success']} (demo alone: {int(np.argmax(d['rewards'][:] > 0))}), "
                  f"{T} frames", flush=True)
        data_out.attrs["total"] = total

        lay = fd.create_group("layout")
        for k, v in layouts.items():
            v = v[kept]
            if k == "robot_qpos":
                v = np.repeat(home[None], len(kept), axis=0)
            lay.create_dataset(k, data=np.asarray(v, np.float64))
        lay.attrs["demos"] = json.dumps([layout_demos[r] for r in kept])
        lay.attrs["home_qpos"] = home
        lay.attrs["home_from_demo"] = layout_demos[home_row]
    env.close()

    steps, errs = np.array(approach_steps), np.array(approach_errs)
    print(f"{len(kept)}/{len(layout_demos)} demos succeed from the home pose -> {out}")
    print(f"approach: {steps.min()}..{steps.max()} steps (mean {steps.mean():.0f}, {fps:.0f} Hz); end-effector "
          f"error to the demo start after it: up to {errs[:, 0].max():.1f} mm, {errs[:, 1].max():.1f} deg")
    if failed:
        print(f"left out (no success from the home pose): {failed}")
    if not args.no_lerobot:
        from resfit.lerobot.dataset.convert_pick_place_to_lerobot import convert_all
        convert_all(out, args.lerobot_dir, name_prefix=args.name)


if __name__ == "__main__":
    main()
