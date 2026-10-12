"""Regenerate the pick-and-place dataset in the simulator, in a new version (the source and
the datasets already built are left as they are), then its LeRobot datasets.

Two independent options:

--home: every episode starts from the same home pose. In the demos the arm starts wherever
  the person's hand was when the video started (the end-effector start varies by up to
  25 cm and 167 deg); a robot starts from a fixed pose. Each demo is preceded by an
  approach from the home pose to the demo's start configuration: interpolated in joint
  space (smoothstep, zero velocity at both ends, so it ends on the demo's own arm
  configuration, with no wrist flip or IK branch change), sent as the end-effector targets
  of those joint positions (forward kinematics), like any action. Its length follows the
  distance: at most APPROACH_SPEED m/s, APPROACH_ROT_SPEED deg/s and APPROACH_JOINT_SPEED
  rad/s on average, at least APPROACH_MIN_STEPS steps; the gripper stays open. The home
  pose is the medoid of the demos' start configurations: the demo start with the smallest
  summed joint-space distance to all the others (a real start, unlike a joint-wise mean).

--canonical-wrist: every demo grasps with the gripper turned the same way about its own
  axis. The demos grasp in two groups 180 deg apart (the wrist joint, joint 7, at about
  +1.4 or -1.8 rad when the gripper closes): the person's hand happened to be turned one way
  or the other. The gripper (two identical fingers) and the bottle (a cylinder) are
  symmetric, so both grasps are the same physically, but the policy sees two behaviours:
  the wrist camera upside down in one group, and rotation targets 180 deg apart for the
  same situation. The demos of the smaller group (wrist at grasp more than 90 deg from the
  median of all demos) are turned by 180 deg about the gripper's own axis for the whole
  episode: every rotation target R becomes R Rz(180 deg) (the tool centre point lies on
  joint 7's axis, its z axis, so positions are unchanged) and the start configuration's
  joint 7 moves by 180 deg (staying within its limits). The same rotation at every step:
  the trajectory keeps its shape.

Each demo is then replayed in the simulator (PickPlaceBottleEnv, the evaluation env): reset
to its layout (bottle, container, table) with the arm at its start (or the home pose), the
approach (--home), its absolute actions (turned with --canonical-wrist), then, if it has
not succeeded yet, its last action held for up to HOLD_STEPS steps. The episode is cut
steps_after_success steps after the first success, as the source dataset is. Everything
the policy gets (observations, both cameras, rewards) is recorded from the simulator, so
the training images are rendered exactly as the evaluation's. A demo that does not reach
success is left out (reported).

Output, next to the source (they share env_scene.xml), named pick_place_bottle[_home]
[_wristcanon]:
  <name>.hdf5  same layout as the source; layout/robot_qpos is each demo's start in this
      version (the home pose for every demo with --home), so the env starts every episode,
      evaluation included, from it. layout.attrs: home_qpos and home_from_demo (--home).
      Per demo, attrs wrist_flipped, approach_steps and approach_end_error_mm / _deg
      (end-effector error to the demo's start after the approach); states are [time, qpos,
      qvel] of env_scene.xml (the source's states are of each demo's own scene); actions_joint
      (T, 8) are the joint targets the IK made of each action, and the gripper (the joint
      action mode's actions).
Then the LeRobot datasets <name>_<mode>/ (convert_pick_place_to_lerobot.py, with its replay
checks and check videos).

Usage, after make_dataset.py:
    uv run resfit/pick_place/regenerate_dataset.py --home                    # pick_place_bottle_home
    uv run resfit/pick_place/regenerate_dataset.py --canonical-wrist         # pick_place_bottle_wristcanon
    uv run resfit/pick_place/regenerate_dataset.py --home --canonical-wrist  # pick_place_bottle_home_wristcanon
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

from resfit.lerobot.dataset.convert_pick_place_to_lerobot import ACTION_MODES  # noqa: E402
from resfit.pick_place import task  # noqa: E402
from resfit.pick_place.env import CAMERA_KEYS, PickPlaceBottleEnv  # noqa: E402

POLICY_DIR = REPO / "videos" / "task" / "dataset_sim" / "policy"
WRIST_JOINT = 6  # joint 7, the last arm joint: its axis is the gripper's z axis
APPROACH_SPEED = 0.15  # m/s, mean end-effector speed of the approach
APPROACH_ROT_SPEED = 90.0  # deg/s, mean end-effector rotation speed
APPROACH_JOINT_SPEED = 1.0  # rad/s, mean speed of the fastest joint
APPROACH_MIN_STEPS = 8
HOLD_STEPS = 8  # steps the last demo action is held for if the demo has not succeeded yet


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--home", action="store_true", help="start every episode from the home pose")
    p.add_argument("--canonical-wrist", action="store_true", help="every demo grasps with the gripper turned the same way")
    p.add_argument("--src", default=str(POLICY_DIR / "pick_place_bottle_base.hdf5"))
    p.add_argument("--lerobot-dir", default=str(REPO / "videos" / "task" / "dataset_sim" / "lerobot"))
    p.add_argument("--modes", nargs="+", default=list(ACTION_MODES), choices=ACTION_MODES,
                   help="LeRobot action modes to write (default: all)")
    p.add_argument("--no-lerobot", action="store_true", help="only write the hdf5")
    args = p.parse_args()
    if not (args.home or args.canonical_wrist):
        p.error("choose --home, --canonical-wrist or both")
    return args


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


def check_wrist_axis(env, fk_data, q_arm):
    """The 180-deg turn of --canonical-wrist is R Rz(180 deg) with unchanged positions only if
    joint 7's axis is the tool centre point's z axis and passes through it."""
    fk_data.qpos[:task.ARM] = q_arm
    mujoco.mj_kinematics(env.model, fk_data)
    mujoco.mj_comPos(env.model, fk_data)
    R = fk_data.site_xmat[env.scene.site].reshape(3, 3)
    axis = R.T @ fk_data.xaxis[WRIST_JOINT]
    offset = R.T @ (fk_data.xanchor[WRIST_JOINT] - fk_data.site_xpos[env.scene.site])
    off_axis = offset - axis * (axis @ offset)
    if np.abs(axis - [0, 0, 1]).max() > 1e-6 or np.linalg.norm(off_axis) > 1e-6:
        raise SystemExit(f"joint 7's axis is not the tool's z axis (axis {axis}, off-axis offset {off_axis})")


def approach_actions(env, fk_data, q_from, q_to, gripper, fps):
    """Absolute base-frame actions (n, 7) moving the arm from q_from to q_to (see --home)."""
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
    return np.array(actions, np.float32).reshape(-1, 7)


def turn_wrist(actions):
    """Absolute actions (n, 7) with the gripper turned by 180 deg about its own z axis."""
    a = np.array(actions, np.float64)
    a[:, 3:6] = (Rotation.from_rotvec(a[:, 3:6]) * Rotation.from_euler("z", np.pi)).as_rotvec()
    return a.astype(np.float32)


def record(env):
    obs = task.obs_to_base(task.world_lowdim_obs(env.data, env.scene), env.pb, env.Rb)
    images = env.get_images(CAMERA_KEYS)
    state = np.concatenate([[env.data.time], env.data.qpos, env.data.qvel])
    return obs, images, state


def rollout(env, fk_data, row, q_start, home, demo_actions, steps_after_success, fps):
    """One demo (see the module docstring): from the home pose with an approach to q_start
    if home is given, else from q_start. Returns the episode's arrays (None if it never
    succeeds) and the approach's length and end error (None without --home)."""
    env.reset(options={"demo": row})
    approach = np.zeros((0, 7), np.float32)
    if home is not None:
        approach = approach_actions(env, fk_data, home[:task.ARM], q_start[:task.ARM], demo_actions[0, 6], fps)
    plan = list(approach) + list(demo_actions) + [demo_actions[-1]] * HOLD_STEPS

    frames, first_success, approach_err = [], None, None
    for t, action in enumerate(plan):
        if home is not None and t == len(approach):
            p, r = eef_base_pose(env, fk_data, q_start[:task.ARM])
            live = task.obs_to_base(task.world_lowdim_obs(env.data, env.scene), env.pb, env.Rb)
            approach_err = (1000 * float(np.linalg.norm(live["robot0_eef_pos"] - p)),
                            float(np.degrees((Rotation.from_quat(live["robot0_eef_quat"]) * r.inv()).magnitude())))
        obs, images, state = record(env)
        _, reward, success, _, _ = env.step(action)
        joint_target = np.concatenate([env.q_target, [action[6]]])  # what the IK made of it
        frames.append((obs, images, state, action, reward, joint_target))
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
        "actions_joint": np.stack([f[5] for f in frames]).astype(np.float32),
        "dones": np.eye(T, dtype=np.int64)[-1],
        "first_success": first_success,
    }
    return episode, len(approach), approach_err


def main():
    args = parse_args()
    src = Path(args.src)
    name = "pick_place_bottle" + ("_home" if args.home else "") + ("_wristcanon" if args.canonical_wrist else "")
    out = src.parent / f"{name}.hdf5"

    with h5py.File(src, "r") as f:
        layout_demos = json.loads(f["layout"].attrs["demos"])
        layouts = {k: f["layout"][k][:] for k in f["layout"]}
        steps_after_success = int(f["data"].attrs["steps_after_success"])
        fps = float(f["data"][layout_demos[0]].attrs["control_freq"])
        demo_actions = {n: f["data"][n]["actions"][:] for n in layout_demos}
        # Wrist joint when the gripper first closes (states rows: [time, qpos, qvel]).
        grasp_wrist = np.array([f["data"][n]["states"][int(np.argmax(demo_actions[n][:, 6] > 0)), 1 + WRIST_JOINT]
                                for n in layout_demos])

    env = PickPlaceBottleEnv(src, camera_names=list(CAMERA_KEYS))
    fk_data = mujoco.MjData(env.model)
    q_starts = layouts["robot_qpos"].copy()
    flipped = np.zeros(len(layout_demos), bool)
    if args.canonical_wrist:
        check_wrist_axis(env, fk_data, q_starts[0][:task.ARM])
        reference = float(np.median(grasp_wrist))
        flipped = np.abs(grasp_wrist - reference) > np.pi / 2
        lo, hi = env.model.jnt_range[WRIST_JOINT]
        for row in np.flatnonzero(flipped):
            q7 = q_starts[row, WRIST_JOINT]
            q_starts[row, WRIST_JOINT] = q7 + np.pi if q7 < reference else q7 - np.pi
            if not lo <= q_starts[row, WRIST_JOINT] <= hi:
                raise SystemExit(f"{layout_demos[row]}: its turned start leaves joint 7's range")
            demo_actions[layout_demos[row]] = turn_wrist(demo_actions[layout_demos[row]])
        print(f"wrist at grasp: median {reference:.2f} rad; {flipped.sum()}/{len(flipped)} demos turned by 180 deg "
              f"about the gripper axis: {[layout_demos[r] for r in np.flatnonzero(flipped)]}", flush=True)

    home, home_row = None, None
    if args.home:
        home_row = medoid(q_starts[:, :task.ARM])
        home = q_starts[home_row].copy()
        p_home, _ = eef_base_pose(env, fk_data, home[:task.ARM])
        print(f"home pose: start of {layout_demos[home_row]} (medoid of the {len(q_starts)} demo starts"
              f"{', turned' if args.canonical_wrist else ''}), end-effector at {np.round(p_home, 3)} m (base frame)",
              flush=True)
    env.layouts["robot_qpos"][:] = home if args.home else q_starts  # where the env's resets start

    kept, failed, approach_steps, approach_errs = [], [], [], []
    with h5py.File(src, "r") as fs, h5py.File(out, "w") as fd:
        data_out = fd.create_group("data")
        for k, v in fs["data"].attrs.items():
            data_out.attrs[k] = v
        total = 0
        for row, demo in enumerate(layout_demos):
            d = fs["data"][demo]
            episode, n_approach, err = rollout(env, fk_data, row, q_starts[row], home, demo_actions[demo],
                                               steps_after_success, fps)
            label = f"  {demo}{' (turned)' if flipped[row] else ''}: "
            if args.home:
                approach_steps.append(n_approach)
                approach_errs.append(err)
                label += f"approach {n_approach} steps (end error {err[0]:.1f} mm, {err[1]:.1f} deg), "
            if episode is None:
                failed.append(demo)
                print(label + "NO SUCCESS: left out", flush=True)
                continue
            g = data_out.create_group(demo)
            for k, v in d.attrs.items():
                g.attrs[k] = v
            T = len(episode["actions"])
            g.attrs["num_samples"] = T
            g.attrs["num_samples_recorded"] = T
            g.attrs["wrist_flipped"] = bool(flipped[row])
            if args.home:
                g.attrs["approach_steps"] = n_approach
                g.attrs["approach_end_error_mm"], g.attrs["approach_end_error_deg"] = err
            for k, v in episode["obs"].items():
                g.create_dataset(f"obs/{k}", data=v)
            for k, v in episode["images"].items():
                g.create_dataset(f"obs/{k}", data=v, compression="gzip", compression_opts=4,
                                 chunks=(1,) + v.shape[1:])
            for k in ("actions", "actions_joint", "rewards", "dones", "states"):
                g.create_dataset(k, data=episode[k])
            kept.append(row)
            total += T
            print(label + f"success at step {episode['first_success']} "
                  f"(source demo: {int(np.argmax(d['rewards'][:] > 0))}), {T} frames", flush=True)
        data_out.attrs["total"] = total

        lay = fd.create_group("layout")
        for k, v in layouts.items():
            if k == "robot_qpos":
                v = np.repeat(home[None], len(layout_demos), axis=0) if args.home else q_starts
            lay.create_dataset(k, data=np.asarray(v[kept], np.float64))
        lay.attrs["demos"] = json.dumps([layout_demos[r] for r in kept])
        if args.home:
            lay.attrs["home_qpos"] = home
            lay.attrs["home_from_demo"] = layout_demos[home_row]
    env.close()

    print(f"{len(kept)}/{len(layout_demos)} demos succeed -> {out}")
    if args.home:
        steps, errs = np.array(approach_steps), np.array(approach_errs)
        print(f"approach: {steps.min()}..{steps.max()} steps (mean {steps.mean():.0f}, {fps:.0f} Hz); end-effector "
              f"error to the demo start after it: up to {errs[:, 0].max():.1f} mm, {errs[:, 1].max():.1f} deg")
    if failed:
        print(f"left out (no success): {failed}")
    if not args.no_lerobot:
        from resfit.lerobot.dataset.convert_pick_place_to_lerobot import convert_all
        convert_all(out, args.lerobot_dir, args.modes, name_prefix=name)


if __name__ == "__main__":
    main()
