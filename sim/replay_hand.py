"""Make the simulated Panda follow the virtual gripper, and render a video.

For every frame, inverse kinematics puts the Panda's tool centre point (between the
fingertip pads) at the virtual gripper position with the same orientation (y =
closing direction, z = approach). It is a kinematic replay: joint angles are set
directly, no physics. IK is damped least squares on the 7 arm joints, warm-started
from the previous frame, with a weak pull towards the home posture to stay away
from odd configurations. A parallel gripper is symmetric under a 180 degree turn
about its approach axis, so the target closest to the current hand is used.

Fingers follow the thumb-index opening minus the finger thickness, clipped to the
Panda's 0-8 cm range.

The bottle (if the scene has one) is carried kinematically: it stands still until
the close, then keeps its pose relative to the gripper while closed, and at the
release it is set down upright on the container floor below where it was let go.

The demo starts and ends with the hand resting flat on the table (used to calibrate
the hand height), which would put the robot in awkward folded poses, and the hand
can be partly out of the image when it starts or stops moving, where the tracking
is unstable. The robot episode is trimmed to the task, on frames where the hand
tracking is trustworthy (thumb / index tips and knuckles inside the image, gripper
orientation not jumping, for --trust-frames frames in a row): from the first such frame where the
gripper is above --lift-height before the grasp (at least --min-approach before it,
otherwise the episode is rejected), to the last such frame after the release.

Video: left, the real video with the rendered robot drawn over it from the phone
camera (gripper and bottle solid, arm and container faint); right, the overview camera with the trajectory
markers. Also saves the joint trajectory (qpos per frame) as JSON.

Usage:
    uv run sim/replay_hand.py videos/task/PXL_20261005_120635999.mp4
"""
import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).parent.parent
HERE = Path(__file__).parent
ARM = 7  # arm joints come first in qpos; then the two finger joints


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="demo video")
    p.add_argument("--scene", default=str(HERE / "scene.xml"), help="scene from build_scene.py")
    p.add_argument("--gripper", default=None,
                   help="default gripper/<video>_gripper_corrected.json (from gripper/depth_correct.py)")
    p.add_argument("--finger-thickness", type=float, default=0.015,
                   help="subtracted from the thumb-index opening to get the gripper gap, m")
    p.add_argument("--lift-height", type=float, default=0.05,
                   help="gripper height above the table that starts / ends the robot episode, m (default 0.05)")
    p.add_argument("--border-margin", type=float, default=20,
                   help="thumb / index tips and knuckles must be this many px inside the image to be trusted "
                        "(default 20)")
    p.add_argument("--max-turn", type=float, default=10,
                   help="max gripper rotation between frames for the tracking to be trusted, deg (default 10)")
    p.add_argument("--trust-frames", type=int, default=5,
                   help="trusted frames in a row needed to start / end the robot episode (default 5)")
    p.add_argument("--min-approach", type=float, default=0.5,
                   help="the robot episode must start at least this long before the grasp, s (default 0.5)")
    p.add_argument("--out", default=None, help="output video (default sim/<video>_robot.mp4)")
    p.add_argument("--no-video", action="store_true", help="only solve and save the joint trajectory")
    return p.parse_args()


def rotation_error(R_target, R):
    """Rotation vector (world frame) taking R to R_target."""
    return cv2.Rodrigues(R_target @ R.T)[0].ravel()


def trusted_frames(gripper, border_margin, max_turn_deg):
    """Per frame: is the hand tracking trustworthy?

    - the keypoints the virtual gripper is built from (thumb and index tips and
      their knuckles) are border_margin px inside the image: if they are cut by
      the border WiLoR guesses them (the wrist or the back of the hand being out
      of the image is fine, it happens all the time with a hand reaching in);
    - the gripper orientation turns less than max_turn_deg since the previous frame:
      real hand motion at 30 fps is much slower, bigger jumps are tracking that has
      not settled (e.g. the hand entering the image).
    """
    hands = json.loads(Path(gripper["hands_json"]).read_text())
    w, h = hands["image_width"], hands["image_height"]
    gripper_joints = [2, 4, 5, 8]  # thumb MCP, thumb tip, index MCP, index tip
    ok = np.zeros(len(gripper["frames"]), bool)
    for i, f in enumerate(hands["frames"]):
        if f["hands"]:
            kp = np.array(f["hands"][0]["keypoints_2d"])[gripper_joints]
            ok[i] = bool(np.all((kp > border_margin) & (kp < np.array([w, h]) - border_margin)))
    R = np.array([f["rotation"] for f in gripper["frames"]])
    turn = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", R[1:], R[:-1]) - 1) / 2, -1, 1)))
    ok[1:] &= turn < max_turn_deg
    return ok


def pose_matrix(pos, mat):
    T = np.eye(4)
    T[:3, :3] = np.asarray(mat).reshape(3, 3)
    T[:3, 3] = pos
    return T


def set_free_pose(data, adr, T):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, T[:3, :3].ravel())
    data.qpos[adr:adr + 3] = T[:3, 3]
    data.qpos[adr + 3:adr + 7] = quat


def solve_ik(model, data, site, p_target, R_target, q_home, iters=100, damping=0.05, rot_weight=0.5):
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    lo, hi = model.jnt_range[:ARM, 0], model.jnt_range[:ARM, 1]
    for _ in range(iters):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        R = data.site_xmat[site].reshape(3, 3)
        e = np.concatenate([p_target - data.site_xpos[site], rot_weight * rotation_error(R_target, R)])
        if np.linalg.norm(e[:3]) < 5e-4 and np.linalg.norm(e[3:]) < rot_weight * np.radians(0.3):
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, site)
        J = np.vstack([jacp[:, :ARM], rot_weight * jacr[:, :ARM]])
        JJt = J @ J.T + damping ** 2 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, e)
        # Posture term in the null space of the task.
        N = np.eye(ARM) - J.T @ np.linalg.solve(JJt, J)
        dq += N @ (0.1 * (q_home - data.qpos[:ARM]))
        data.qpos[:ARM] = np.clip(data.qpos[:ARM] + dq, lo, hi)
    mujoco.mj_kinematics(model, data)
    R = data.site_xmat[site].reshape(3, 3)
    return (np.linalg.norm(p_target - data.site_xpos[site]),
            np.degrees(np.linalg.norm(rotation_error(R_target, R))))


def main():
    args = parse_args()
    stem = Path(args.video).stem
    gripper = json.loads(Path(args.gripper or ROOT / "gripper" / f"{stem}_gripper_corrected.json").read_text())
    out_path = Path(args.out) if args.out else HERE / f"{stem}_robot.mp4"

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "panda/tcp")
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "panda/home")
    mujoco.mj_resetDataKeyframe(model, data, home)
    q_home = data.qpos[:ARM].copy()
    flip = np.diag([-1.0, -1.0, 1.0])  # 180 degrees about the approach axis

    # Trim the episode (see the docstring): the resting hand, and the frames where
    # the hand tracking is not trustworthy yet / any more.
    frames = gripper["frames"]
    fps = gripper["fps"]
    n_all = len(frames)
    z = np.array([f["position"][2] for f in frames])
    events = {e["event"]: e["frame"] for e in gripper["events"]}
    close, release = events["close"], events.get("open", n_all - 1)
    idx = np.arange(n_all)
    trusted = trusted_frames(gripper, args.border_margin, args.max_turn)
    run = args.trust_frames
    above = z > args.lift_height
    # First frame from which the hand is trusted for `run` frames in a row; last
    # frame up to which it was trusted for `run` frames in a row.
    starts = [i for i in range(n_all - run + 1) if above[i] and i <= close and trusted[i:i + run].all()]
    ends = [i for i in range(run - 1, n_all) if above[i] and i >= release and trusted[i - run + 1:i + 1].all()]
    latest_start = close - int(round(args.min_approach * fps))
    if not starts or starts[0] > latest_start:
        raise SystemExit(f"hand tracking not trustworthy early enough before the grasp (frame {close}): "
                         f"no {run} trusted frames in a row by frame {latest_start}")
    first = starts[0]
    last = ends[-1] if ends else release
    n_untrusted = int((~trusted[first:last + 1]).sum())
    frames = frames[first:last + 1]
    print(f"robot episode: frames {first}-{last} ({first / fps:.2f}-{last / fps:.2f} s) of 0-{n_all - 1}; "
          f"grasp at {close}, release at {release}; untrusted frames inside: {n_untrusted}")

    # Bottle: free joint, carried kinematically (see the docstring).
    bottle_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle")
    if bottle_jnt >= 0:
        b_adr = model.jnt_qposadr[bottle_jnt]
        b_start = pose_matrix(model.qpos0[b_adr:b_adr + 3], np.eye(3))  # upright, as built
        half_height = b_start[2, 3]  # stands on the table at z = height / 2
        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "container_floor")
        floor_top = model.geom_pos[fid][2] + model.geom_size[fid][2] if fid >= 0 else 0.0
        b_pose, b_rel, was_closed = b_start.copy(), None, False

    # IK for every frame.
    qpos, pos_err, rot_err = [], [], []
    for f in frames:
        p, R = np.array(f["position"]), np.array(f["rotation"])
        if not qpos:
            # First frame: the home pose says nothing about which of the two
            # symmetric gripper orientations suits the arm, and a poor pick gets stuck
            # against a joint limit, then flips (a big joint jump) a frame later. Solve
            # both and keep the more accurate one (the closer to home if they tie).
            best = None
            for R_try in (R, R @ flip):
                data.qpos[:ARM] = q_home
                ep, er = solve_ik(model, data, site, p, R_try, q_home, iters=500)
                score = (round(ep, 3), np.linalg.norm(data.qpos[:ARM] - q_home))
                if best is None or score < best[0]:
                    best = (score, data.qpos[:ARM].copy(), ep, er)
            data.qpos[:ARM] = best[1]
            ep, er = best[2], best[3]
        else:
            mujoco.mj_kinematics(model, data)
            R_now = data.site_xmat[site].reshape(3, 3)
            if np.linalg.norm(rotation_error(R @ flip, R_now)) < np.linalg.norm(rotation_error(R, R_now)):
                R = R @ flip
            ep, er = solve_ik(model, data, site, p, R, q_home)
        finger = np.clip((f["opening"] - args.finger_thickness) / 2, 0.0, 0.04)
        data.qpos[ARM:ARM + 2] = finger
        if bottle_jnt >= 0:
            mujoco.mj_kinematics(model, data)
            T_tcp = pose_matrix(data.site_xpos[site], data.site_xmat[site])
            if f["closed"]:
                if b_rel is None:  # grasp: freeze the bottle relative to the gripper
                    b_rel = np.linalg.inv(T_tcp) @ b_pose
                b_pose = T_tcp @ b_rel
            elif was_closed:  # release: set it down upright on the container floor
                b_pose = pose_matrix([b_pose[0, 3], b_pose[1, 3], floor_top + half_height], np.eye(3))
            was_closed = f["closed"]
            set_free_pose(data, b_adr, b_pose)
        qpos.append(data.qpos.copy())
        pos_err.append(ep)
        rot_err.append(er)
    pos_err, rot_err = np.array(pos_err) * 1000, np.array(rot_err)
    print(f"IK over {len(qpos)} frames: position error median {np.median(pos_err):.1f} mm "
          f"(max {pos_err.max():.1f}), orientation error median {np.median(rot_err):.1f} deg "
          f"(max {rot_err.max():.1f})")

    traj_path = out_path.with_suffix(".json")
    traj_path.write_text(json.dumps({
        "video": args.video, "scene": args.scene, "fps": fps,
        "frames": [f["frame"] for f in frames],  # video frame of each qpos row
        "joint_names": [model.joint(i).name for i in range(model.njnt)],
        "qpos": np.round(qpos, 5).tolist(),
        "closed": [f["closed"] for f in frames],
        "ik_position_error_mm": np.round(pos_err, 2).tolist(),
        "ik_orientation_error_deg": np.round(rot_err, 2).tolist(),
    }))

    if args.no_video:
        print(f"Saved {traj_path}")
        return
    renderer = OverlayRenderer(model)
    cap = open_video(args.video, skip=first)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, renderer.size)
    for i, q in enumerate(qpos):
        ok, real = cap.read()
        if not ok:
            break
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        f = frames[i]
        text = f"frame {f['frame']}  t={f['time_s']:.2f}s  IK error {pos_err[i]:.1f} mm, {rot_err[i]:.1f} deg"
        writer.write(renderer.render(data, real, f["closed"], text))
    cap.release()
    writer.release()
    renderer.close()
    print(f"Saved {out_path}\nSaved {traj_path}")


def open_video(path, skip=0):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    for _ in range(skip):
        cap.read()
    return cap


class OverlayRenderer:
    """Video frame: left, the real frame with the sim drawn over it from the phone
    camera; right, the overview camera.

    Phone view: gripper and bottle solid (they should land on the real hand and
    bottle), arm and container faint. The arm sits between the camera and the
    gripper, so the gripper is drawn in a second pass with the arm's visual geoms
    moved to group 5 and hidden.
    """

    def __init__(self, model):
        gripper_bodies = {"panda/hand", "panda/left_finger", "panda/right_finger"}
        self.arm_alpha, self.gripper_alpha = np.zeros(model.ngeom), np.zeros(model.ngeom)
        for g in range(model.ngeom):
            name = model.body(model.geom_bodyid[g]).name
            if name in gripper_bodies or name == "bottle":
                self.gripper_alpha[g] = 0.85
            elif name == "container":
                self.gripper_alpha[g] = 0.35
            elif name.startswith("panda/"):
                self.arm_alpha[g] = 0.25
                if model.geom_group[g] == 2:
                    model.geom_group[g] = 5
        W, H = model.vis.global_.offwidth, model.vis.global_.offheight
        model.vis.global_.offwidth = max(W, 1280)
        self.phone_opt = mujoco.MjvOption()
        self.phone_opt.geomgroup[1] = False  # phone marker
        self.phone_opt.geomgroup[5] = True   # arm
        self.gripper_opt = mujoco.MjvOption()
        self.gripper_opt.geomgroup[1] = False
        self.gripper_opt.geomgroup[5] = False
        self.over_opt = mujoco.MjvOption()
        self.over_opt.geomgroup[5] = True
        # The phone view is shown at 540x960, so render it at that size directly
        # (same view: the camera intrinsics scale with the image).
        self.phone = mujoco.Renderer(model, 960, 540)
        self.over = mujoco.Renderer(model, 960, 1280)
        self.size = (540 + 1280, 960)

    def render(self, data, real, closed, text):
        left = cv2.resize(real, (540, 960), interpolation=cv2.INTER_AREA).astype(np.float32)
        for opt, alpha in ((self.phone_opt, self.arm_alpha), (self.gripper_opt, self.gripper_alpha)):
            self.phone.update_scene(data, camera="phone", scene_option=opt)
            sim = cv2.cvtColor(self.phone.render(), cv2.COLOR_RGB2BGR)
            self.phone.enable_segmentation_rendering()
            seg = self.phone.render()
            self.phone.disable_segmentation_rendering()
            is_geom = seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM
            a = np.zeros(is_geom.shape)
            a[is_geom] = alpha[seg[:, :, 0][is_geom]]
            a = a[:, :, None]
            left = (1 - a) * left + a * sim
        left = left.astype(np.uint8)

        self.over.update_scene(data, camera="overview", scene_option=self.over_opt)
        right = cv2.cvtColor(self.over.render(), cv2.COLOR_RGB2BGR)

        state, color = ("CLOSED", (0, 0, 230)) if closed else ("OPEN", (0, 170, 0))
        cv2.rectangle(left, (10, 10), (190, 70), color, -1)
        cv2.putText(left, state, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(right, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return np.hstack([left, right])

    def close(self):
        self.phone.close()
        self.over.close()


if __name__ == "__main__":
    main()
