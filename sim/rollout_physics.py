"""Physics rollout: the Panda replays the recorded joint trajectory, physics does the rest.

Only the initial state is set by hand: the robot at the first recorded pose with
the gripper open, the bottle standing where it was picked (from the scene), the
container fixed. From then on MuJoCo simulates everything:

  arm     : the Panda's position actuators track the recorded joint angles
            (sim/<video>_robot.json from replay_hand.py), linearly interpolated
            between video frames
  gripper : binary command from the open/closed state: fully closed while closed
            (it stops on the bottle and squeezes), fully open otherwise
  bottle  : free body; it moves only through contact, friction and gravity

Success ("the bottle is in the box"), checked at every step and at the end:
  - the bottle touches the container (its floor or the inside of its walls),
  - the bottle's centre is inside the container's inner footprint,
  - the bottle no longer touches the gripper (it has been let go).
The bottle does not need to be upright; its final tilt is reported.

With --record, every step is also saved to an HDF5 file laid out like one
robomimic demo (obs / next_obs / actions / rewards / dones / states, see
pipeline/make_robomimic.py), with sim camera images from a fixed camera facing
the robot across the table (agentview) and a wrist camera (eye_in_hand).

The video shows one frame per recorded frame next to the real video, so with a
slowdown the simulation is played back faster than it ran.

Usage:
    uv run sim/rollout_physics.py videos/task/PXL_20261005_120635999.mp4
"""
import argparse
import json
from pathlib import Path

import cv2
import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from replay_hand import OverlayRenderer, open_video

HERE = Path(__file__).parent
ARM = 7
GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0  # actuator8 ctrl: 255 = 8 cm open, 0 = closed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="demo video")
    p.add_argument("--scene", default=str(HERE / "scene.xml"), help="scene from build_scene.py")
    p.add_argument("--robot", default=None, help="joint trajectory, default sim/<video>_robot.json")
    p.add_argument("--slowdown", type=float, default=2.0,
                   help="play the trajectory this many times slower than the demo (default 2: at 1x the "
                        "wrist would need ~290 deg/s, about twice a real Panda's joint speed limit)")
    p.add_argument("--hold", type=float, default=0.5,
                   help="keep simulating this long after the last frame, s (default 0.5)")
    p.add_argument("--out", default=None, help="output video (default sim/<video>_physics.mp4)")
    p.add_argument("--out-json", default=None, help="outcome + states JSON (default: next to the video)")
    p.add_argument("--record", default=None, help="save the episode as one robomimic-style demo (HDF5)")
    p.add_argument("--image-size", type=int, default=128,
                   help="agentview and eye-in-hand image size, square (default 128)")
    p.add_argument("--no-video", action="store_true", help="skip the overlay video")
    return p.parse_args()


class Scene:
    """Ids and helpers for the bodies the rollout cares about."""

    def __init__(self, model):
        self.m = model
        body = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        self.bottle, self.cont = body("bottle"), body("container")
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "panda/tcp")
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "container_floor")
        wall = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "container_wall_px")
        self.inner_half = np.array([model.geom_pos[wall][0] - model.geom_size[wall][0],
                                    model.geom_size[floor][1] - model.geom_size[wall][0]])
        owner = [model.body(model.geom_bodyid[g]).name for g in range(model.ngeom)]
        self.bottle_geoms = {g for g, n in enumerate(owner) if n == "bottle"}
        self.box_geoms = {g for g, n in enumerate(owner) if n == "container"}
        self.robot_geoms = {g for g, n in enumerate(owner) if n.startswith("panda/")}

    def in_box(self, data):
        """Success test (see the module docstring)."""
        touch_box = touch_robot = False
        for i in range(data.ncon):
            g1, g2 = data.contact[i].geom1, data.contact[i].geom2
            if g1 in self.bottle_geoms or g2 in self.bottle_geoms:
                other = g2 if g1 in self.bottle_geoms else g1
                touch_box |= other in self.box_geoms
                touch_robot |= other in self.robot_geoms
        rel = data.xpos[self.bottle][:2] - data.xpos[self.cont][:2]
        inside = bool(np.all(np.abs(rel) < self.inner_half))
        return touch_box and inside and not touch_robot, inside, touch_box, touch_robot

    def tilt_deg(self, data):
        return float(np.degrees(np.arccos(np.clip(data.xmat[self.bottle][8], -1, 1))))


def quat_xyzw(mat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat).ravel())
    return q[[1, 2, 3, 0]]  # robosuite / robomimic convention


class Recorder:
    """Observations, actions and states per step, saved as one robomimic demo."""

    def __init__(self, model, scene, size):
        self.m, self.s = model, scene
        self.agent = mujoco.Renderer(model, size, size)
        self.wrist = mujoco.Renderer(model, size, size)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[1] = False      # phone marker
        self.opt.sitegroup[:] = False      # trajectory markers, tcp site
        self.fk = mujoco.MjData(model)     # for the target end-effector pose of an action
        self.obs, self.actions, self.actions_abs, self.actions_joint = [], [], [], []
        self.states, self.rewards = [], []

    def observe(self, data):
        site, b = self.s.site, self.s.bottle
        eef_pos = data.site_xpos[site].copy()
        o = {
            "robot0_joint_pos": data.qpos[:ARM].copy(),
            "robot0_joint_vel": data.qvel[:ARM].copy(),
            "robot0_eef_pos": eef_pos,
            "robot0_eef_quat": quat_xyzw(data.site_xmat[site]),
            "robot0_gripper_qpos": data.qpos[ARM:ARM + 2].copy(),
            "robot0_gripper_qvel": data.qvel[ARM:ARM + 2].copy(),
            # bottle position, quaternion (xyzw), bottle relative to the gripper, container position
            "object": np.concatenate([data.xpos[b], quat_xyzw(data.xmat[b]), data.xpos[b] - eef_pos,
                                      data.xpos[self.s.cont]]),
        }
        self.agent.update_scene(data, camera="agentview", scene_option=self.opt)
        o["agentview_image"] = self.agent.render().copy()
        self.wrist.update_scene(data, camera="panda/wrist", scene_option=self.opt)
        o["robot0_eye_in_hand_image"] = self.wrist.render().copy()
        self.obs.append(o)
        self.states.append(np.concatenate([[data.time], data.qpos, data.qvel]))

    def act(self, data, q_target, closed):
        """Action for the coming interval: the target the controller is driven to."""
        site = self.s.site
        g = 1.0 if closed else -1.0
        self.fk.qpos[:] = data.qpos
        self.fk.qpos[:ARM] = q_target
        mujoco.mj_kinematics(self.m, self.fk)
        t_pos, t_R = self.fk.site_xpos[site].copy(), self.fk.site_xmat[site].reshape(3, 3).copy()
        R = data.site_xmat[site].reshape(3, 3)
        d_rot = Rotation.from_matrix(t_R @ R.T).as_rotvec()
        self.actions.append(np.concatenate([t_pos - data.site_xpos[site], d_rot, [g]]))
        self.actions_abs.append(np.concatenate([t_pos, Rotation.from_matrix(t_R).as_rotvec(), [g]]))
        self.actions_joint.append(np.concatenate([q_target, [g]]))

    def save(self, path, attrs):
        """obs has one more entry than actions (the final observation): next_obs = obs[1:]."""
        T = len(self.actions)
        with h5py.File(path, "w") as f:
            for k, v in attrs.items():
                f.attrs[k] = v
            f.attrs["num_samples"] = T
            f.create_dataset("actions", data=np.array(self.actions, dtype=np.float32))
            f.create_dataset("actions_abs", data=np.array(self.actions_abs, dtype=np.float32))
            f.create_dataset("actions_joint", data=np.array(self.actions_joint, dtype=np.float32))
            f.create_dataset("rewards", data=np.array(self.rewards, dtype=np.float32))
            dones = np.zeros(T, dtype=np.int64)
            dones[-1] = 1
            f.create_dataset("dones", data=dones)
            f.create_dataset("states", data=np.array(self.states[:T], dtype=np.float64))
            for group, sl in (("obs", slice(0, T)), ("next_obs", slice(1, T + 1))):
                for key in self.obs[0]:
                    arr = np.stack([o[key] for o in self.obs[sl]])
                    if arr.dtype == np.uint8:
                        f.create_dataset(f"{group}/{key}", data=arr, compression="gzip", compression_opts=4,
                                         chunks=(1,) + arr.shape[1:])
                    else:
                        f.create_dataset(f"{group}/{key}", data=arr.astype(np.float32))

    def close(self):
        self.agent.close()
        self.wrist.close()


def main():
    args = parse_args()
    stem = Path(args.video).stem
    traj = json.loads(Path(args.robot or HERE / f"{stem}_robot.json").read_text())
    out_path = Path(args.out) if args.out else HERE / f"{stem}_physics.mp4"

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    scene = Scene(model)
    q_arm = np.array(traj["qpos"])[:, :ARM]
    closed = np.array(traj["closed"])
    fps = traj["fps"]
    n = len(q_arm)
    dt_frame = args.slowdown / fps  # sim time per recorded frame
    steps_per_frame = int(round(dt_frame / model.opt.timestep))

    # Initial state: scene defaults (bottle on the table), robot at the first pose, open.
    mujoco.mj_resetData(model, data)
    data.qpos[:ARM] = q_arm[0]
    data.qpos[ARM:ARM + 2] = 0.04
    data.ctrl[:ARM] = q_arm[0]
    data.ctrl[ARM] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)
    bottle_start = data.xpos[scene.bottle].copy()
    rec = Recorder(model, scene, args.image_size) if args.record else None

    # Simulate; keep one state per recorded frame (+ hold) for the video.
    states, bottle_z, track_err = [], [], []
    n_hold = int(round(args.hold / dt_frame))
    for k in range(n + n_hold):
        k0 = min(k, n - 1)
        k1 = min(k + 1, n - 1)
        if rec:
            rec.observe(data)
            rec.act(data, q_arm[k1], bool(closed[k0]))
        data.ctrl[ARM] = GRIPPER_CLOSED if closed[k0] else GRIPPER_OPEN
        for s in range(steps_per_frame):
            a = s / steps_per_frame
            data.ctrl[:ARM] = (1 - a) * q_arm[k0] + a * q_arm[k1]
            mujoco.mj_step(model, data)
        if rec:
            rec.rewards.append(float(scene.in_box(data)[0]))
        states.append(data.qpos.copy())
        bottle_z.append(data.xpos[scene.bottle][2])
        track_err.append(np.abs(data.qpos[:ARM] - q_arm[k1]).max())
    if rec:
        rec.observe(data)  # final observation (next_obs of the last step)

    # Outcome.
    success, inside, touch_box, touch_robot = scene.in_box(data)
    end = data.xpos[scene.bottle]
    rel = end[:2] - data.xpos[scene.cont][:2]
    tilt = scene.tilt_deg(data)
    lifted = max(bottle_z) - bottle_start[2]
    print(f"{n} recorded frames, {model.opt.timestep * 1000:g} ms steps, slowdown {args.slowdown:g}")
    print(f"arm tracking error: max {np.degrees(max(track_err)):.1f} deg")
    print(f"bottle lifted up to {100 * lifted:.1f} cm above its start")
    print(f"bottle end: {np.round(100 * end, 1).tolist()} cm, {np.round(100 * rel, 1).tolist()} cm from the "
          f"container centre, tilt {tilt:.0f} deg ({'upright' if tilt < 20 else 'tipped over'})")
    print(f"success: {success} (inside footprint {inside}, touching the container {touch_box}, "
          f"touching the gripper {touch_robot})")

    result = {
        "video": args.video, "robot": traj.get("scene"), "slowdown": args.slowdown,
        "success": success,
        "inside_container": inside,
        "touching_container": touch_box,
        "touching_gripper": touch_robot,
        "bottle_lifted_m": round(float(lifted), 4),
        "bottle_end_m": np.round(end, 4).tolist(),
        "bottle_tilt_deg": round(float(tilt), 1),
        "arm_tracking_error_max_deg": round(float(np.degrees(max(track_err))), 1),
        "qpos": np.round(states, 5).tolist(),
    }
    json_path = Path(args.out_json) if args.out_json else out_path.with_suffix(".json")
    json_path.write_text(json.dumps(result))
    print(f"Saved {json_path}")

    if rec:
        rec.save(args.record, {
            "clip": str(args.video),
            "model_file": Path(args.scene).read_text(),
            "control_freq": fps / args.slowdown,
            "success": success,
            "bottle_tilt_deg": tilt,
        })
        rec.close()
        print(f"Saved {args.record} ({len(rec.actions)} samples)")

    if args.no_video:
        return
    # Video, aligned with the recorded frames (the last real frame is held at the end).
    renderer = OverlayRenderer(model)
    cap = open_video(args.video, skip=traj["frames"][0])
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, renderer.size)
    real = None
    for k, q in enumerate(states):
        if k < n:
            ok, frame = cap.read()
            real = frame if ok else real
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        frame_id = traj["frames"][min(k, n - 1)]
        text = (f"PHYSICS  frame {frame_id}  bottle z {100 * data.xpos[scene.bottle][2]:.1f} cm"
                + ("  (hold)" if k >= n else ""))
        writer.write(renderer.render(data, real, bool(closed[min(k, n - 1)]), text))
    cap.release()
    writer.release()
    renderer.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
