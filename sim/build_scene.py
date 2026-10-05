"""Build the MuJoCo scene: table, Panda, virtual tag, and the phone camera.

World frame = tag frame: origin at the tag centre, x/y on the table top, z up,
metres. So the table top is z = 0 and every trajectory from the earlier steps
(hand keypoints, virtual gripper) is already in world coordinates.

  table   : box whose top is at z = 0
  tag     : textured plane with the same ArUco marker and white margin as the print
  camera  : "phone", at the camera pose from tag_pose (T_tag_cam), with the
            calibrated intrinsics (fx, fy, cx, cy; lens distortion is ignored,
            it is below a pixel over most of the image)
  robot   : Menagerie Franka Panda, base on the table plane. Default (--base-mode
            behind): --base-distance (0.5 m) behind the midpoint between the grasp
            and the release (towards the camera, i.e. roughly where the person's
            body was), facing the workspace (+y). Referenced to the workspace, not
            to the camera, so it adapts to how close the person sat. A scan over
            positions around the workspace showed this is where the IK can follow
            a side-on human pinch (1 mm median error); to the right or diagonally,
            the hand orientation is out of the Panda's wrist range.
            Other modes: right (--base-distance to the right, facing -x) and
            shoulder (below the camera, 10 cm in front of it, facing +y).
  markers : optional small spheres along the virtual gripper trajectory
            (green = open, red = closed), to see it in the overlay
  objects : from the virtual gripper events, not from vision. The episode has a
            single grasp, so the bottle stands where the gripper closed on it
            (x/y of the gripper over the first closed frames), and the container
            is centred where the gripper released it, a bit larger than the real
            one so the bottle lands inside despite a few cm of tracking error.
            The bottle has a free joint (body "bottle") so it can be simulated.

Check: render from the "phone" camera, blend with the real first frame, and detect
the ArUco tag in the render; its corners should land on the real tag's corners.

Usage:
    uv run sim/build_scene.py videos/task/PXL_20261005_120635999.mp4
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).parent.parent
HERE = Path(__file__).parent
PANDA_XML = HERE / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"
MENAGERIE = "https://github.com/google-deepmind/mujoco_menagerie"

# Panda reach: the wrist centre can be ~0.78 m from the shoulder joint (0.333 m
# above the base); the wrist centre is ~0.21 m behind the fingertip centre.
SHOULDER_HEIGHT, WRIST_TO_TCP, WRIST_REACH = 0.333, 0.21, 0.78
# Cameras relative to the robot base, for a robot facing +y (turned with the base yaw).
# Agentview (dataset): across the table, facing the robot. Overview: three-quarter view.
AGENTVIEW_EYE, AGENTVIEW_TARGET, AGENTVIEW_FOVY = np.array([0.0, 1.25, 0.8]), np.array([0.0, 0.35, 0.25]), 50
OVERVIEW_EYE, OVERVIEW_TARGET = np.array([1.1, 0.9, 1.0]), np.array([0.0, 0.45, 0.15])
WRIST_CAM_POS, WRIST_CAM_TARGET = [0.05, 0, 0.065], [0, 0, 0.16]  # in the Panda hand frame


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="demo video (used for the overlay and to find the step outputs)")
    p.add_argument("--intrinsics", default=str(ROOT / "calibration" / "camera_intrinsics.json"))
    p.add_argument("--tag-pose", default=None, help="default tag_pose/<video>_tag_pose.json")
    p.add_argument("--gripper", default=None,
                   help="default gripper/<video>_gripper_corrected.json (from gripper/depth_correct.py)")
    p.add_argument("--tag-size", type=float, default=None, help="tag side in m (default: from the tag pose)")
    p.add_argument("--tag-margin", type=float, default=0.020, help="white margin around the tag, m")
    p.add_argument("--base-mode", choices=["behind", "right", "shoulder"], default="behind",
                   help="automatic robot placement (default behind, see the module docstring)")
    p.add_argument("--base-distance", type=float, default=0.5,
                   help="base-mode behind / right: distance from the workspace centre, m (default 0.5)")
    p.add_argument("--base", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="explicit robot base position in the tag frame, m (overrides --base-mode)")
    p.add_argument("--base-yaw", type=float, default=None,
                   help="robot base yaw, degrees (default: facing the workspace: 90 = +y for behind and "
                        "shoulder, 180 = -x for right)")
    p.add_argument("--bottle", type=float, nargs=2, default=[0.035, 0.075], metavar=("DIAMETER", "HEIGHT"),
                   help="pill bottle cylinder, m (default 0.035 0.075)")
    p.add_argument("--box-size", type=float, nargs=3, default=[0.16, 0.14, 0.07], metavar=("X", "Y", "HEIGHT"),
                   help="container inner size and wall height, m (default 0.16 0.14 0.07)")
    p.add_argument("--out-dir", default=str(HERE), help="where to write scene.xml and the overlay images")
    return p.parse_args()


def ensure_panda():
    if PANDA_XML.exists():
        return
    import subprocess
    dest = HERE / "third_party" / "mujoco_menagerie"
    print(f"Downloading the Panda model from {MENAGERIE} ...")
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", MENAGERIE, str(dest)],
                   check=True)
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "franka_emika_panda"], check=True)


def make_tag_texture(path, tag_id, tag_size, margin, px_per_m=4000):
    """Same marker as tag/generate_tag.py: DICT_5X5_50, white margin around it."""
    tag_px = 7 * round(tag_size * px_per_m / 7)  # 5x5 tag = 7x7 cells
    margin_px = round(margin * tag_px / tag_size)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    img = cv2.aruco.generateImageMarker(d, tag_id, tag_px)
    img = cv2.copyMakeBorder(img, margin_px, margin_px, margin_px, margin_px, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(str(path), img)
    return (tag_px + 2 * margin_px) / tag_px * tag_size  # printed sheet side, m


def object_positions(gripper, n_frames=6):
    """Pick and place from the gripper: median over the first closed frames (the
    bottle is still on the table) and over the last frames before release. Use the
    depth-corrected gripper (gripper/depth_correct.py): uncorrected monocular depth
    puts these points several cm off along the camera ray."""
    events = {e["event"]: e["frame"] for e in gripper["events"]}
    P = np.array([f["position"] for f in gripper["frames"]])
    close, release = events["close"], events["open"]
    pick = np.median(P[close:close + n_frames], axis=0)
    place = np.median(P[max(close, release - n_frames):release + 1], axis=0)
    return pick, place


def add_objects(spec, pick, place, bottle, box):
    world = spec.worldbody
    radius, height = bottle[0] / 2, bottle[1]
    cap = 0.015
    body = world.add_body(name="bottle", pos=[pick[0], pick[1], height / 2])
    body.add_freejoint(name="bottle")
    # condim 4 adds torsional friction: without it a bottle held by the cap pivots
    # freely about the line between the two fingertips and swings out of the grasp.
    body.add_geom(name="bottle_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[radius, (height - cap) / 2, 0],
                  pos=[0, 0, -cap / 2], rgba=[0.18, 0.12, 0.1, 1], mass=0.05, condim=4)
    body.add_geom(name="bottle_cap", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[radius * 1.03, cap / 2, 0],
                  pos=[0, 0, (height - cap) / 2], rgba=[0.95, 0.95, 0.95, 1], mass=0.005, condim=4)

    # Open container: floor + 4 walls, inner size box[0] x box[1], walls box[2] high.
    lx, ly, wall_h = box
    t = 0.003  # wall thickness
    rgba = [0.85, 0.88, 0.9, 0.6]
    cont = world.add_body(name="container", pos=[place[0], place[1], 0])
    cont.add_geom(name="container_floor", type=mujoco.mjtGeom.mjGEOM_BOX, size=[lx / 2 + t, ly / 2 + t, t / 2],
                  pos=[0, 0, t / 2], rgba=rgba)
    for name, size, pos in [
        ("container_wall_px", [t / 2, ly / 2 + t, wall_h / 2], [lx / 2 + t / 2, 0, wall_h / 2]),
        ("container_wall_nx", [t / 2, ly / 2 + t, wall_h / 2], [-lx / 2 - t / 2, 0, wall_h / 2]),
        ("container_wall_py", [lx / 2, t / 2, wall_h / 2], [0, ly / 2 + t / 2, wall_h / 2]),
        ("container_wall_ny", [lx / 2, t / 2, wall_h / 2], [0, -ly / 2 - t / 2, wall_h / 2]),
    ]:
        cont.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, rgba=rgba)


def look_at(eye, target, up=(0, 0, 1)):
    """Quaternion of a MuJoCo camera at eye looking at target, with the image up towards `up`."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.stack([right, up, -forward], axis=1).ravel())
    return quat


def cv_to_mujoco_camera(R_cv):
    """OpenCV camera axes (x right, y down, z forward) -> MuJoCo (x right, y up, looks along -z)."""
    return R_cv @ np.diag([1.0, -1.0, -1.0])


def build(args, T_tag_cam, calib, tag_id, tag_size, gripper):
    spec = mujoco.MjSpec()
    spec.modelname = "pick_place_scene"
    spec.compiler.degree = False
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # same as the Panda model
    # Grasp-friendly contact solver (objects, friction coefficients and grip force are
    # unchanged). With MuJoCo's default (pyramidal friction cones, impratio 1) the
    # held bottle vibrates in the fingers (~300 deg/s) and creeps along the pads
    # until it drops. Elliptic cones with a high impratio make friction stiff, as
    # MuJoCo recommends for grasping: the swing falls to ~15 deg/s.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    # Paths in the saved scene.xml are relative to its folder.
    out_dir = Path(args.out_dir).resolve()
    spec.modelfiledir = str(out_dir)
    spec.meshdir = os.path.relpath(PANDA_XML.parent / "assets", out_dir)
    w, h = calib["image_width"], calib["image_height"]
    spec.visual.global_.offwidth = w
    spec.visual.global_.offheight = h
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]

    world = spec.worldbody
    world.add_light(pos=[0, -0.5, 2.0], dir=[0, 0.2, -1], diffuse=[0.6, 0.6, 0.6])

    # Table: top at z = 0, covering the tag, the trajectory and the robot base.
    base = np.array(args.base)
    x0, x1 = min(-0.4, base[0] - 0.4), max(0.6, base[0] + 0.4)
    y0, y1 = min(-0.7, base[1] - 0.15), 0.4
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[(x1 - x0) / 2, (y1 - y0) / 2, 0.02], pos=[(x0 + x1) / 2, (y0 + y1) / 2, -0.02],
                   rgba=[0.85, 0.85, 0.82, 1])

    # Tag: a plane with the marker texture, a hair above the table to avoid z-fighting.
    sheet = make_tag_texture(out_dir / "tag_texture.png", tag_id, tag_size, args.tag_margin)
    spec.add_texture(name="tag", type=mujoco.mjtTexture.mjTEXTURE_2D, file="tag_texture.png")
    mat = spec.add_material(name="tag")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tag"
    mat.texrepeat = [1, 1]
    mat.texuniform = False
    world.add_geom(name="tag", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[sheet / 2, sheet / 2, 0.01],
                   pos=[0, 0, 0.0005], material="tag", contype=0, conaffinity=0)

    # Phone camera: pose from the tag pose, intrinsics from the calibration.
    K = np.array(calib["camera_matrix"])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, cv_to_mujoco_camera(T_tag_cam[:3, :3]).ravel())
    cam = world.add_camera(name="phone", pos=T_tag_cam[:3, 3], quat=quat)
    cam.resolution = [w, h]
    cam.sensor_size = [w, h]                 # 1 sensor unit = 1 pixel
    cam.focal_pixel = [K[0, 0], K[1, 1]]
    # Principal point as an offset from the image centre. MuJoCo's sign convention
    # is the opposite of OpenCV's on both axes (checked with the tag overlay).
    cam.principal_pixel = [w / 2 - K[0, 2], h / 2 - K[1, 2]]
    # A visual marker for the phone (visual only, group 1 so it does not show in the phone view).
    world.add_geom(name="phone_marker", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.035, 0.075, 0.004],
                   pos=T_tag_cam[:3, 3], quat=quat, rgba=[0.15, 0.15, 0.15, 1], group=1,
                   contype=0, conaffinity=0)

    # Cameras placed relative to the robot (offsets given for a robot facing +y, then
    # turned with the base yaw). Overview: three-quarter side view of the scene.
    turn = np.radians(args.base_yaw - 90)
    Rz = np.array([[np.cos(turn), -np.sin(turn), 0], [np.sin(turn), np.cos(turn), 0], [0, 0, 1]])
    eye = base + Rz @ OVERVIEW_EYE
    world.add_camera(name="overview", pos=eye, quat=look_at(eye, base + Rz @ OVERVIEW_TARGET), fovy=50)
    # Agentview camera for the robot dataset (robomimic "agentview_image"): across
    # the table, facing the robot, so the arm does not hide the gripper and objects
    # (from the phone viewpoint, the robot at the shoulder covers most of the image).
    eye = base + Rz @ AGENTVIEW_EYE
    world.add_camera(name="agentview", pos=eye, quat=look_at(eye, base + Rz @ AGENTVIEW_TARGET),
                     fovy=AGENTVIEW_FOVY)

    # Robot.
    ensure_panda()
    panda = mujoco.MjSpec.from_file(str(PANDA_XML))
    # Tool centre point between the fingertip pads, with the hand's axes: y = finger
    # sliding (closing) direction, z = approach. Same convention as the virtual gripper.
    panda.body("hand").add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.005, 0, 0], group=4)
    # Wrist camera (robomimic "eye_in_hand"): on the side of the hand housing, looking
    # down along the fingers, past the fingertips (placed below the housing so it does
    # not fill the image).
    panda.body("hand").add_camera(name="wrist", pos=WRIST_CAM_POS, quat=look_at(WRIST_CAM_POS, WRIST_CAM_TARGET,
                                                                                up=[1, 0, 0]), fovy=75)
    # Grip force. Menagerie's gripper is a soft position servo (100 N/m on the finger
    # tendon): closing on a 3.5 cm object squeezes with only ~2 N, and the bottle
    # slides in the fingers. 10x the stiffness gives ~17 N (a real Panda grips with
    # up to 70 N); ctrl 255 still means fully open.
    grip = panda.actuator("actuator8")
    grip.gainprm[0] *= 10
    grip.biasprm[1] *= 10
    grip.biasprm[2] *= 10
    yaw = np.radians(args.base_yaw)
    frame = world.add_frame(pos=base, quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    spec.attach(panda, prefix="panda/", frame=frame)

    # Objects and gripper trajectory markers (visual only).
    if gripper is not None:
        pick, place = object_positions(gripper)
        add_objects(spec, pick[:2], place[:2], args.bottle, args.box_size)
        for f in gripper["frames"][::3]:
            world.add_site(pos=f["position"], size=[0.006, 0, 0], group=0,
                           rgba=[0.85, 0.1, 0.1, 1] if f["closed"] else [0.1, 0.7, 0.2, 1])
    return spec


def render(model, camera, home_key, show_robot=True, show_group1=False, size=None):
    """Render from a camera. The Panda's visual meshes are geom group 2; group 1
    holds the phone marker, which must not show up in the phone's own view."""
    data = mujoco.MjData(model)
    if home_key is not None:
        mujoco.mj_resetDataKeyframe(model, data, home_key)
    mujoco.mj_forward(model, data)
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = show_group1
    opt.geomgroup[2] = show_robot
    h, w = size or (model.vis.global_.offheight, model.vis.global_.offwidth)
    with mujoco.Renderer(model, h, w) as r:
        r.update_scene(data, camera=camera, scene_option=opt)
        return cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)


def detect_tag(img, tag_id):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50), params)
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if ids is None or tag_id not in ids.ravel():
        return None
    return corners[list(ids.ravel()).index(tag_id)].reshape(4, 2)


def main():
    args = parse_args()
    stem = Path(args.video).stem
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag_pose = json.loads(Path(args.tag_pose or ROOT / "tag_pose" / f"{stem}_tag_pose.json").read_text())
    calib = json.loads(Path(args.intrinsics).read_text())
    T_tag_cam = np.array(tag_pose["T_tag_cam"])
    tag_size = args.tag_size or tag_pose["tag_size_m"]
    gripper_path = Path(args.gripper or ROOT / "gripper" / f"{stem}_gripper_corrected.json")
    gripper = json.loads(gripper_path.read_text()) if gripper_path.exists() else None

    cam_pos = T_tag_cam[:3, 3]
    mode = "explicit" if args.base is not None else args.base_mode
    if mode in ("behind", "right") and gripper is None:
        print("WARNING: no gripper file to locate the workspace; robot placed with --base-mode shoulder")
        mode = "shoulder"
    if mode in ("behind", "right"):
        pick, place = object_positions(gripper)
        centre = (pick + place) / 2
        if mode == "behind":
            args.base = [centre[0], centre[1] - args.base_distance, 0.0]
        else:
            args.base = [centre[0] + args.base_distance, centre[1], 0.0]
    elif mode == "shoulder":
        args.base = [cam_pos[0], cam_pos[1] + 0.10, 0.0]
    if args.base_yaw is None:
        args.base_yaw = 180.0 if mode == "right" else 90.0
    print(f"camera position (tag frame): {np.round(cam_pos, 3).tolist()} m")
    print(f"robot base position        : {np.round(args.base, 3).tolist()} m, yaw {args.base_yaw:g} deg ({mode})")

    if gripper is not None:
        P = np.array([f["position"] for f in gripper["frames"]])
        approach = np.array([f["rotation"] for f in gripper["frames"]])[:, :, 2]
        shoulder = np.array(args.base) + [0, 0, SHOULDER_HEIGHT]
        reach = np.linalg.norm(P - WRIST_TO_TCP * approach - shoulder, axis=1)
        print(f"reach check: wrist centre {reach.min():.2f}-{reach.max():.2f} m from the shoulder "
              f"(Panda limit ~{WRIST_REACH} m)" + ("  WARNING: out of reach" if reach.max() > WRIST_REACH else ""))
        pick, place = object_positions(gripper)
        print(f"gripper file: {gripper_path.name}" + ("" if "depth_correction" in gripper else
              "  WARNING: not depth-corrected, run gripper/depth_correct.py"))
        print(f"bottle    : x/y {np.round(100 * pick[:2], 1).tolist()} cm (gripper close, at height "
              f"{100 * pick[2]:.1f} cm; bottle is {100 * args.bottle[1]:g} cm tall), standing on the table")
        print(f"container : x/y {np.round(100 * place[:2], 1).tolist()} cm (gripper release, at height "
              f"{100 * place[2]:.1f} cm), inner {100 * args.box_size[0]:g} x {100 * args.box_size[1]:g} cm")

    spec = build(args, T_tag_cam, calib, tag_pose["tag_id"], tag_size, gripper)
    xml_path = out_dir / "scene.xml"
    xml_path.write_text(spec.to_xml())
    model = mujoco.MjModel.from_xml_path(str(xml_path))  # check the saved file loads on its own
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "panda/home")
    print(f"Saved {xml_path} ({model.nbody} bodies, {model.nq} dofs)")

    # Overlay check on the first frame of the video (the tag pose was averaged there).
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    ok, real = cap.read()
    cap.release()
    # The robot sits at the shoulder, so in its home pose (arm up) it fills the
    # phone's view: hide it for the tag check.
    home = home if home >= 0 else None
    sim = render(model, "phone", home, show_robot=False)
    blend = cv2.addWeighted(real, 0.5, sim, 0.5, 0)
    overview = render(model, "overview", home, show_group1=True, size=(810, 1080))
    cv2.imwrite(str(out_dir / f"{stem}_sim_overview.png"), overview)

    real_c, sim_c = detect_tag(real, tag_pose["tag_id"]), detect_tag(sim, tag_pose["tag_id"])
    if sim_c is None:
        print("WARNING: the tag was not detected in the render")
    else:
        err = np.linalg.norm(real_c - sim_c, axis=1)
        print(f"tag corners, render vs real: mean {err.mean():.2f} px, max {err.max():.2f} px")
        for c, color in ((real_c, (0, 255, 0)), (sim_c, (255, 0, 255))):
            cv2.polylines(blend, [c.astype(np.int32)], True, color, 2, cv2.LINE_AA)
        cv2.putText(blend, f"tag: real (green) vs sim (magenta), mean error {err.mean():.1f} px", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.imwrite(str(out_dir / f"{stem}_sim_render.png"), sim)
    cv2.imwrite(str(out_dir / f"{stem}_sim_overlay.png"), blend)
    for name in ("render", "overlay", "overview"):
        print(f"Saved {out_dir / f'{stem}_sim_{name}.png'}")


if __name__ == "__main__":
    main()
