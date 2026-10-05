"""Estimate the camera pose relative to the ArUco tag in a demo video.

The pose is averaged over the first still second (the corners are averaged across
those frames, then a single PnP is solved). An annotated video is written to check
the result: the averaged tag axes should stick to the tag.

Tag frame (OpenCV IPPE_SQUARE convention): origin at the tag centre, x to the right
and y up as the tag is printed, z out of the paper (up from the table). Units: metres.

Usage:
    uv run tag_pose/estimate_tag_pose.py videos/task/PXL_20261005_120635999.mp4 --tag-size 150
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
HERE = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="demo video")
    p.add_argument("--intrinsics", default=str(ROOT / "calibration" / "camera_intrinsics.json"))
    p.add_argument("--tag-size", type=float, default=150.0, help="measured side of the black square, mm (default 150)")
    p.add_argument("--tag-id", type=int, default=0)
    p.add_argument("--dict", default="DICT_5X5_50", help="ArUco dictionary (default DICT_5X5_50)")
    p.add_argument("--still-seconds", type=float, default=1.0, help="length of the still window at the start")
    p.add_argument("--out", default=None, help="output JSON (default tag_pose/<video>_tag_pose.json)")
    p.add_argument("--no-video", action="store_true", help="skip the annotated video (keep its first frame)")
    p.add_argument("--out-video", default=None, help="annotated video (default tag_pose/<video>_axes.mp4)")
    return p.parse_args()


def to_matrix(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T


def solve_pose(corners, obj_points, K, dist):
    ok, rvec, tvec = cv2.solvePnP(obj_points, corners, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners) ** 2, axis=1))))
    return rvec, tvec, err


def main():
    args = parse_args()
    stem = Path(args.video).stem
    out_json = Path(args.out) if args.out else HERE / f"{stem}_tag_pose.json"
    out_video = Path(args.out_video) if args.out_video else HERE / f"{stem}_axes.mp4"

    calib = json.loads(Path(args.intrinsics).read_text())
    K = np.array(calib["camera_matrix"])
    dist = np.array(calib["dist_coeffs"])

    s = args.tag_size / 1000.0  # metres
    # ArUco corner order: top-left, top-right, bottom-right, bottom-left.
    obj_points = np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0], [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]])

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict)), params)

    # Same decoding as the calibration: apply the phone's rotation metadata.
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Pass 1: detect the tag in every frame.
    frames, detections = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        found = None
        if ids is not None and args.tag_id in ids.ravel():
            found = corners[list(ids.ravel()).index(args.tag_id)].reshape(4, 2).astype(np.float64)
        frames.append(frame)
        detections.append(found)
    cap.release()

    h, w = frames[0].shape[:2]
    if (w, h) != (calib["image_width"], calib["image_height"]):
        raise SystemExit(f"Video frames are {w}x{h} but intrinsics are for "
                         f"{calib['image_width']}x{calib['image_height']}")
    n_detected = sum(d is not None for d in detections)
    print(f"{len(frames)} frames at {fps:.1f} fps, {w}x{h}; tag {args.tag_id} detected in {n_detected}")

    # Average over the first still second.
    n_still = int(round(args.still_seconds * fps))
    still = [d for d in detections[:n_still] if d is not None]
    if len(still) < max(3, n_still // 2):
        raise SystemExit(f"Tag detected in only {len(still)} of the first {n_still} frames; "
                         "it must be fully visible at the start of the video.")
    still = np.array(still)
    motion = float(np.max(np.linalg.norm(still - still.mean(axis=0), axis=2)))

    rvec, tvec, err = solve_pose(still.mean(axis=0), obj_points, K, dist)
    T_cam_tag = to_matrix(rvec, tvec)          # tag frame -> camera frame
    T_tag_cam = np.linalg.inv(T_cam_tag)       # camera pose expressed in the tag frame

    # Spread of the per-frame poses in the still window, as a stability check.
    per_frame = [solve_pose(c, obj_points, K, dist) for c in still]
    t_all = np.array([p[1].ravel() for p in per_frame])
    rot_dev = [np.degrees(np.linalg.norm(cv2.Rodrigues(cv2.Rodrigues(p[0])[0] @ T_cam_tag[:3, :3].T)[0]))
               for p in per_frame]

    cam_pos = T_tag_cam[:3, 3]
    result = {
        "video": str(args.video),
        "intrinsics": str(args.intrinsics),
        "tag_id": args.tag_id,
        "tag_dict": args.dict,
        "tag_size_m": s,
        "frames_averaged": len(still),
        "still_window_s": args.still_seconds,
        "T_cam_tag": T_cam_tag.tolist(),
        "T_tag_cam": T_tag_cam.tolist(),
        "rvec_cam_tag": rvec.ravel().tolist(),
        "tvec_cam_tag_m": tvec.ravel().tolist(),
        "camera_position_in_tag_m": cam_pos.tolist(),
        "reprojection_error_px": err,
        "still_window_corner_motion_px": motion,
        "per_frame_translation_std_m": t_all.std(axis=0).tolist(),
        "per_frame_rotation_dev_max_deg": float(np.max(rot_dev)),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))

    # Pass 2: annotated video. Thick axes = averaged pose (fixed); green outline =
    # per-frame detection. If the camera stays still, the axes stay on the outline.
    # With --no-video only the first annotated frame is saved (as a PNG).
    axis_len = s / 2
    writer = None if args.no_video else cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i, (frame, det) in enumerate(zip(frames, detections)):
        img = frame.copy()
        if det is not None:
            cv2.polylines(img, [det.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawFrameAxes(img, K, dist, rvec, tvec, axis_len, 4)
        label = "still window (averaged)" if i < n_still else ""
        cv2.putText(img, f"frame {i}  {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        if i == 0:
            cv2.imwrite(str(out_video.with_suffix(".png")), img)
        if writer is None:
            break
        writer.write(img)
    if writer is not None:
        writer.release()

    print("\n=== Tag pose (averaged over the first "
          f"{args.still_seconds:g} s, {len(still)} frames) ===")
    print(f"tag size            : {args.tag_size:g} mm")
    print(f"tag in camera (t)   : {np.round(tvec.ravel(), 4).tolist()} m, distance {np.linalg.norm(tvec):.3f} m")
    print(f"camera in tag frame : {np.round(cam_pos, 4).tolist()} m (height above tag plane {cam_pos[2]:.3f} m)")
    print(f"reprojection error  : {err:.2f} px")
    print(f"corner motion in still window: {motion:.2f} px", "(WARNING: camera or tag moved)" if motion > 2 else "")
    print(f"per-frame spread    : translation std {np.round(t_all.std(axis=0) * 1000, 1).tolist()} mm, "
          f"max rotation deviation {np.max(rot_dev):.2f} deg")
    print(f"\nSaved {out_json}\nSaved " + (f"{out_video.with_suffix('.png')}" if args.no_video else
                                           f"{out_video} (and {out_video.with_suffix('.png').name})"))


if __name__ == "__main__":
    main()
