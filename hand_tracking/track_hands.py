"""Track 3D hand keypoints in a demo video with WiLoR (via wilor-mini).

For every frame: detect hands, run WiLoR, and express the 21 hand keypoints
(OpenPose order) in the camera frame using the calibrated intrinsics, and in the
tag frame if a tag pose is available. An overlay video is written to check that
the keypoints follow the fingers.

Monocular depth comes from the hand's apparent size, so it is only as good as
WiLoR's guess of the hand's size. With a tag pose, the hand scale is calibrated on
the rest periods at the start and end of the video, where the hand is assumed to
lie flat on the table (detected as the hand barely moving in the image).

Usage:
    uv run hand_tracking/track_hands.py videos/task/PXL_20261005_120635999.mp4
"""
import argparse
import json
import os
import pickle
from pathlib import Path

# torch >= 2.6 loads checkpoints with weights_only=True by default, which the WiLoR
# checkpoint and ultralytics' YOLO hand detector (full pickled models) do not
# support. Both come from fixed, trusted Hugging Face repos: restore the old default.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

ROOT = Path(__file__).parent.parent
HERE = Path(__file__).parent

JOINT_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 4, 8, 9
KNUCKLES = [5, 9, 13, 17]
PALM = [0, 5, 9, 13, 17]  # wrist + knuckles
FINGERS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
FINGER_COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0)]  # BGR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="demo video")
    p.add_argument("--intrinsics", default=str(ROOT / "calibration" / "camera_intrinsics.json"))
    p.add_argument("--tag-pose", default=None,
                   help="tag pose JSON (default tag_pose/<video>_tag_pose.json if it exists)")
    p.add_argument("--out", default=None, help="output JSON (default hand_tracking/<video>_hands.json)")
    p.add_argument("--no-video", action="store_true", help="skip the overlay video")
    p.add_argument("--out-video", default=None, help="overlay video (default hand_tracking/<video>_hands.mp4)")
    p.add_argument("--conf", type=float, default=0.3, help="hand detector confidence threshold")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                   "mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--hand-scale", type=float, default=None,
                   help="fixed hand scale factor (default: calibrate on the first still second)")
    p.add_argument("--rest-speed", type=float, default=3.0,
                   help="max palm speed in the image (px/frame) for the hand to count as resting")
    p.add_argument("--rest-min-seconds", type=float, default=0.3,
                   help="shortest rest period used for the hand-scale calibration")
    p.add_argument("--palm-height", type=float, default=0.02,
                   help="height of the wrist/knuckle joints above the table when the hand lies flat, m")
    return p.parse_args()


def prepare_mano(model_dir):
    """Download MANO_RIGHT.pkl if needed and strip its chumpy objects.

    The pickle stores two arrays as chumpy objects, and chumpy does not work with
    numpy 2. Unpickle them as plain stubs and replace them with numpy arrays.
    """
    path = model_dir / "pretrained_models" / "MANO_RIGHT.pkl"
    if not path.exists():
        hf_hub_download(repo_id="warmshao/WiLoR-mini", subfolder="pretrained_models",
                        filename="MANO_RIGHT.pkl", local_dir=model_dir)

    class Stub:
        def __setstate__(self, state):
            self.__dict__.update(state)

    class StubUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("chumpy"):
                return type(name, (Stub,), {})
            return super().find_class(module, name)

    with open(path, "rb") as f:
        data = StubUnpickler(f, encoding="latin1").load()
    stubs = [k for k, v in data.items() if isinstance(v, Stub)]
    if not stubs:
        return
    for k in stubs:
        v = data[k]
        if hasattr(v, "idxs"):  # chumpy Select: a view into another chumpy array
            data[k] = np.asarray(v.a.x).ravel()[v.idxs].reshape(v.preferred_shape)
        else:                   # chumpy Ch: plain array
            data[k] = np.asarray(v.x)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Converted chumpy arrays {stubs} in {path} to numpy")


def place_hand(kp3d_rel, kp2d, bbox, K, dist):
    """Place WiLoR's 3D hand in the real camera frame so it projects onto kp2d.

    WiLoR predicts the hand orientation as seen in its crop, as if the crop were
    straight ahead of the camera (near-orthographic, focal 5000 px). With the real
    wide-angle camera the hand is seen along the ray through the box centre, so
    rotate it by the angle between the optical axis and that ray. Then solve the
    translation by least squares: each joint, in normalised coordinates (x, y),
    gives tx - x tz = x Z - X and ty - y tz = y Z - Y.
    """
    centre = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    ray = np.append(cv2.undistortPoints(centre.reshape(1, 1, 2), K, dist).ravel(), 1.0)
    ray /= np.linalg.norm(ray)
    axis = np.cross([0.0, 0.0, 1.0], ray)
    R = cv2.Rodrigues(axis / np.linalg.norm(axis) * np.arcsin(np.linalg.norm(axis)))[0]
    kp3d = kp3d_rel @ R.T

    xy = cv2.undistortPoints(kp2d.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    X, Y, Z = kp3d.T
    x, y = xy.T
    ones, zeros = np.ones_like(x), np.zeros_like(x)
    A = np.concatenate([np.stack([ones, zeros, -x], 1), np.stack([zeros, ones, -y], 1)])
    b = np.concatenate([x * Z - X, y * Z - Y])
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    return kp3d + t


def overlap(a, b):
    """Intersection over the smaller box: catches a box nested inside another."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))


def draw_hand(img, kp2d, is_right, thumb_index_mm):
    pts = kp2d.astype(int)
    for finger, color in zip(FINGERS, FINGER_COLORS):
        for a, b in zip(finger[:-1], finger[1:]):
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 3, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), 5, (255, 255, 255), -1, cv2.LINE_AA)
    for i in KNUCKLES:
        cv2.rectangle(img, tuple(pts[i] - 7), tuple(pts[i] + 7), (255, 0, 255), 2)
    for i in (THUMB_TIP, INDEX_TIP):
        cv2.circle(img, tuple(pts[i]), 12, (255, 0, 255), 3, cv2.LINE_AA)
    cv2.line(img, tuple(pts[THUMB_TIP]), tuple(pts[INDEX_TIP]), (255, 0, 255), 2, cv2.LINE_AA)
    label = f"{'R' if is_right else 'L'} thumb-index {thumb_index_mm:.0f} mm"
    cv2.putText(img, label, (pts[0][0] - 100, pts[0][1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)


def find_rest_frames(per_frame, K, dist, max_speed, min_len):
    """Frames of the rest periods touching the start and end of the video.

    A frame is still when the palm centre moves less than max_speed px from the
    previous frame (image motion does not depend on the hand scale). Keep the run
    of still frames that starts at the first frame and the one that ends at the
    last frame, if they last at least min_len frames.
    """
    centres = []
    for hands in per_frame:
        if hands:
            c, _ = cv2.projectPoints(hands[0][2][PALM], np.zeros(3), np.zeros(3), K, dist)
            centres.append(c.reshape(-1, 2).mean(axis=0))
        else:
            centres.append(None)
    still = [False] + [a is not None and b is not None and np.linalg.norm(a - b) < max_speed
                       for a, b in zip(centres[:-1], centres[1:])]
    still[0] = still[1]
    n = len(still)
    start = 0
    while start < n and still[start]:
        start += 1
    end = n
    while end > 0 and still[end - 1]:
        end -= 1
    rest = []
    if start >= min_len:
        rest += list(range(start))
    if n - end >= min_len:
        rest += [i for i in range(end, n) if i >= start]
    return rest


def format_ranges(frames):
    if not frames:
        return "none"
    runs, first = [], frames[0]
    for a, b in zip(frames, frames[1:] + [None]):
        if b != a + 1:
            runs.append(f"{first}-{a}")
            first = b
    return ", ".join(runs)


def open_video(path):
    # Same decoding as the calibration: apply the phone's rotation metadata.
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return cap


def main():
    args = parse_args()
    stem = Path(args.video).stem
    out_json = Path(args.out) if args.out else HERE / f"{stem}_hands.json"
    out_video = Path(args.out_video) if args.out_video else HERE / f"{stem}_hands.mp4"
    tag_pose_path = Path(args.tag_pose) if args.tag_pose else ROOT / "tag_pose" / f"{stem}_tag_pose.json"

    calib = json.loads(Path(args.intrinsics).read_text())
    K = np.array(calib["camera_matrix"])
    dist = np.array(calib["dist_coeffs"])
    T_tag_cam = None
    if tag_pose_path.exists():
        T_tag_cam = np.array(json.loads(tag_pose_path.read_text())["T_tag_cam"])
        print(f"Using tag pose {tag_pose_path}")
    else:
        print("No tag pose found; keypoints are only given in the camera frame")

    prepare_mano(HERE)
    pipe = WiLorHandPose3dEstimationPipeline(device=torch.device(args.device), dtype=torch.float32,
                                             wilor_pretrained_dir=str(HERE), verbose=False)
    # WiLoR feeds some convolutions permuted/sliced (non-contiguous) tensors, which
    # conv2d rejects on MPS. Make every conv input contiguous.
    for m in pipe.wilor_model.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_pre_hook(lambda _, inputs: tuple(x.contiguous() for x in inputs))

    # Pass 1: run WiLoR on every frame.
    cap = open_video(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    per_frame = []  # list of hands per frame: (is_right, bbox, kp3d_cam unscaled)
    residuals = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if (w, h) != (calib["image_width"], calib["image_height"]):
            raise SystemExit(f"Video frames are {w}x{h} but intrinsics are for "
                             f"{calib['image_width']}x{calib['image_height']}")
        hands = []
        for det in pipe.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), hand_conf=args.conf):
            # The detector sometimes returns the same hand twice (as left and right);
            # detections come sorted by confidence, so keep the first of overlapping boxes.
            if any(overlap(det["hand_bbox"], b) > 0.5 for _, b, _ in hands):
                continue
            preds = det["wilor_preds"]
            kp2d_wilor = preds["pred_keypoints_2d"][0]
            kp3d_cam = place_hand(preds["pred_keypoints_3d"][0], kp2d_wilor, det["hand_bbox"], K, dist)
            kp2d, _ = cv2.projectPoints(kp3d_cam, np.zeros(3), np.zeros(3), K, dist)
            residuals.append(np.mean(np.linalg.norm(kp2d.reshape(-1, 2) - kp2d_wilor, axis=1)))
            hands.append((bool(det["is_right"]), det["hand_bbox"], kp3d_cam))
        per_frame.append(hands)
        print(f"\rframe {len(per_frame)}/{n_total}  hands: {len(hands)}", end="", flush=True)
    cap.release()
    print()

    rest_frames = find_rest_frames(per_frame, K, dist, args.rest_speed, int(round(args.rest_min_seconds * fps)))
    print(f"Rest frames (hand flat on the table): {format_ranges(rest_frames)}")

    # Hand scale. Scaling the hand about the camera centre keeps its image
    # projection, and only changes its size and distance. Pick the scale that puts
    # the palm at --palm-height above the table in the rest frames:
    # height(s) = r3 . (s P) + t3, averaged over the palm joints.
    scale = 1.0
    if args.hand_scale is not None:
        scale = args.hand_scale
    elif T_tag_cam is not None:
        r3, t3 = T_tag_cam[2, :3], T_tag_cam[2, 3]
        scales = [(args.palm_height - t3) / np.mean(per_frame[i][0][2][PALM] @ r3) for i in rest_frames]
        if scales:
            scale = float(np.median(scales))
            print(f"Hand scale calibrated on {len(scales)} rest frames: {scale:.3f} "
                  f"(spread {np.std(scales):.3f})")
        else:
            print("WARNING: no rest period found at the start or end; hand scale left at 1.0")

    # Pass 2: write results and (unless --no-video) the overlay video.
    frames_out = []
    cap = writer = None
    if not args.no_video:
        cap = open_video(args.video)
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for idx, hands in enumerate(per_frame):
        frame = cap.read()[1] if cap is not None else None
        hands_out = []
        for is_right, bbox, kp3d_cam in hands:
            kp3d_cam = scale * kp3d_cam
            kp2d, _ = cv2.projectPoints(kp3d_cam, np.zeros(3), np.zeros(3), K, dist)
            kp2d = kp2d.reshape(-1, 2)
            hand = {
                "is_right": is_right,
                "bbox": [round(v, 1) for v in bbox],
                "keypoints_2d": kp2d.round(2).tolist(),
                "keypoints_3d_cam": kp3d_cam.round(5).tolist(),
            }
            if T_tag_cam is not None:
                kp3d_tag = kp3d_cam @ T_tag_cam[:3, :3].T + T_tag_cam[:3, 3]
                hand["keypoints_3d_tag"] = kp3d_tag.round(5).tolist()
            hands_out.append(hand)
            if frame is not None:
                thumb_index_mm = 1000 * np.linalg.norm(kp3d_cam[THUMB_TIP] - kp3d_cam[INDEX_TIP])
                draw_hand(frame, kp2d, is_right, thumb_index_mm)
        frames_out.append({"frame": idx, "time_s": round(idx / fps, 4), "hands": hands_out})
        if frame is not None:
            cv2.putText(frame, f"frame {idx}  hands: {len(hands)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 0, 255), 3)
            writer.write(frame)
    if writer is not None:
        cap.release()
        writer.release()

    result = {
        "video": str(args.video),
        "fps": fps,
        "image_width": w,
        "image_height": h,
        "intrinsics": str(args.intrinsics),
        "tag_pose": str(tag_pose_path) if T_tag_cam is not None else None,
        "hand_scale": scale,
        "rest_frames": rest_frames,
        "joint_names": JOINT_NAMES,
        "units": "metres; keypoints_3d_cam in the OpenCV camera frame, keypoints_3d_tag in the tag frame",
        "frames": frames_out,
    }
    out_json.write_text(json.dumps(result))

    with_hand = sum(1 for hands in per_frame if hands)
    palm = [np.linalg.norm(hands[0][2][0] - hands[0][2][MIDDLE_MCP]) * scale for hands in per_frame if hands]
    print(f"\nHands found in {with_hand}/{len(per_frame)} frames")
    if residuals:
        print(f"Mean 2D residual between WiLoR keypoints and the placed 3D hand: {np.mean(residuals):.1f} px")
    print(f"Hand scale {scale:.3f}: wrist to middle knuckle = {1000 * np.median(palm):.0f} mm "
          "(measure yours to check)")
    print(f"Saved {out_json}" + ("" if args.no_video else f"\nSaved {out_video}"))


if __name__ == "__main__":
    main()
