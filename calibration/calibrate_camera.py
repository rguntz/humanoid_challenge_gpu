"""Calibrate phone camera intrinsics from a checkerboard video.

Usage:
    uv run calibration/calibrate_camera.py videos/lens/PXL_20261005_112858764.mp4
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="checkerboard calibration video")
    p.add_argument("--cols", type=int, default=9, help="inner corners per row (default 9)")
    p.add_argument("--rows", type=int, default=6, help="inner corners per column (default 6)")
    p.add_argument("--square", type=float, default=20.0, help="square size in mm (default 20)")
    p.add_argument("--out", default=str(HERE / "camera_intrinsics.json"), help="output JSON path")
    p.add_argument("--debug-dir", default=None, help="debug image folder (default: <out dir>/calib_debug)")
    p.add_argument("--step", type=int, default=3, help="try every Nth frame (default 3)")
    p.add_argument("--max-frames", type=int, default=40, help="frames to keep for calibration (default 40)")
    p.add_argument("--free-k3", action="store_true",
                   help="also estimate k3 (off by default: it overfits unless the board reaches the image corners)")
    return p.parse_args()


def open_video(path):
    # Let OpenCV apply the phone's rotation metadata, so frames come out upright,
    # exactly as the demo videos will be decoded later.
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return cap


def detect_corners(gray, pattern):
    ok, corners = cv2.findChessboardCornersSB(gray, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if not ok:
            return None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    corners = corners.reshape(-1, 2)
    # The detector may return the grid in either direction; make the first corner
    # the top-left one so the tilt features below are comparable across frames.
    if corners[0].sum() > corners[-1].sum():
        corners = corners[::-1].copy()
    return corners


def board_features(corners, pattern, w, h):
    """Position, size and tilt of the board, used to pick diverse frames."""
    cols, rows = pattern
    grid = corners.reshape(rows, cols, 2)
    tl, tr, bl, br = grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1]
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    cx, cy = corners.mean(axis=0)
    area = cv2.contourArea(np.array([tl, tr, br, bl], dtype=np.float32))
    return np.array([
        cx / w,
        cy / h,
        np.sqrt(area / (w * h)),                    # apparent board size (distance)
        (right - left) / (right + left),            # tilt around the vertical axis
        (bottom - top) / (bottom + top),            # tilt around the horizontal axis
    ])


def sharpness(gray, corners):
    """Variance of the Laplacian inside the board's bounding box."""
    x, y, bw, bh = cv2.boundingRect(corners.astype(np.float32))
    roi = gray[max(y, 0):y + bh, max(x, 0):x + bw]
    return cv2.Laplacian(roi, cv2.CV_64F).var()


def select_diverse(features, n):
    """Greedy farthest-point sampling in feature space."""
    # Tilt features are small numbers; scale them up so they count as much as position.
    weights = np.array([1.0, 1.0, 1.0, 3.0, 3.0])
    f = features * weights
    chosen = [int(np.argmin(np.linalg.norm(f[:, :2] - 0.5, axis=1)))]  # start near the centre
    dist = np.linalg.norm(f - f[chosen[0]], axis=1)
    while len(chosen) < min(n, len(f)):
        i = int(np.argmax(dist))
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(f - f[i], axis=1))
    return sorted(chosen)


def calibrate(obj_points, img_points, size, flags):
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None, flags=flags)
    errors = []
    for objp, imgp, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        errors.append(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - imgp.reshape(-1, 2)) ** 2, axis=1))))
    return rms, K, dist, np.array(errors)


def corner_coverage(points, w, h):
    """Which of the four image corners (outer 20% x 20% regions) contain detected corners."""
    names = {"top-left": (0, 0), "top-right": (1, 0), "bottom-left": (0, 1), "bottom-right": (1, 1)}
    covered = {}
    for name, (right, bottom) in names.items():
        in_x = points[:, 0] > 0.8 * w if right else points[:, 0] < 0.2 * w
        in_y = points[:, 1] > 0.8 * h if bottom else points[:, 1] < 0.2 * h
        covered[name] = int(np.sum(in_x & in_y))
    return covered


def main():
    args = parse_args()
    pattern = (args.cols, args.rows)
    out_path = Path(args.out)
    debug_dir = Path(args.debug_dir) if args.debug_dir else out_path.parent / "calib_debug"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # 1-2. Decode frames and detect corners on every Nth frame.
    cap = open_video(args.video)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detections = []  # (frame index, corners, features, sharpness)
    w = h = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if h is None:
            h, w = frame.shape[:2]
        if idx % args.step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = detect_corners(gray, pattern)
            if corners is not None:
                detections.append((idx, corners, board_features(corners, pattern, w, h), sharpness(gray, corners)))
            print(f"\rframe {idx}/{n_total}  detections: {len(detections)}", end="", flush=True)
        idx += 1
    cap.release()
    print(f"\nDecoded frame size: {w}x{h}, board found in {len(detections)} of {idx // args.step + 1} tried frames")
    if len(detections) < 10:
        raise SystemExit("Too few detections to calibrate.")

    # 3. Reject blurry frames (keep the sharpest quarter: a hand-held board is often
    # motion-blurred, and blur is the main source of reprojection error), then pick
    # a diverse subset.
    sharp = np.array([d[3] for d in detections])
    threshold = np.percentile(sharp, 75)
    detections = [d for d in detections if d[3] >= threshold]
    print(f"After blur rejection (Laplacian var >= {threshold:.0f}): {len(detections)} frames")
    chosen = select_diverse(np.array([d[2] for d in detections]), args.max_frames)
    selected = [detections[i] for i in chosen]
    print(f"Selected {len(selected)} diverse frames")

    # 4. Calibrate, drop outlier frames, recalibrate.
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square
    obj_points = [objp] * len(selected)
    img_points = [d[1].reshape(-1, 1, 2).astype(np.float32) for d in selected]

    flags = 0 if args.free_k3 else cv2.CALIB_FIX_K3
    rms, K, dist, errors = calibrate(obj_points, img_points, (w, h), flags)
    print(f"\nInitial RMS: {rms:.3f} px")
    for d, e in zip(selected, errors):
        print(f"  frame {d[0]:5d}: {e:.3f} px")

    limit = max(2.0 * np.median(errors), 0.3)
    keep = errors <= limit
    if not keep.all() and keep.sum() >= 10:
        dropped = [d[0] for d, k in zip(selected, keep) if not k]
        print(f"Removing {len(dropped)} outlier frames (error > {limit:.3f} px): {dropped}")
        selected = [d for d, k in zip(selected, keep) if k]
        obj_points = [objp] * len(selected)
        img_points = [d[1].reshape(-1, 1, 2).astype(np.float32) for d in selected]
        rms, K, dist, errors = calibrate(obj_points, img_points, (w, h), flags)
        print(f"Recalibrated RMS: {rms:.3f} px")

    # 5. Save intrinsics.
    result = {
        "video": str(args.video),
        "image_width": w,
        "image_height": h,
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.ravel().tolist(),
        "rms_error_px": float(rms),
        "num_frames": len(selected),
        "pattern_inner_corners": [args.cols, args.rows],
        "square_size_mm": args.square,
    }
    out_path.write_text(json.dumps(result, indent=2))

    # 6. Debug images. Re-read the video once to grab the selected frames.
    wanted = {d[0] for d in selected}
    frames = {}
    cap = open_video(args.video)
    idx = 0
    while len(frames) < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            frames[idx] = frame
        idx += 1
    cap.release()

    for f in debug_dir.glob("*.png"):
        f.unlink()
    for d in selected[:: max(1, len(selected) // 6)][:6]:
        img = frames[d[0]].copy()
        cv2.drawChessboardCorners(img, pattern, d[1].reshape(-1, 1, 2).astype(np.float32), True)
        cv2.imwrite(str(debug_dir / f"corners_frame{d[0]:05d}.png"), img)

    all_points = np.concatenate([d[1] for d in selected])
    coverage = cv2.addWeighted(frames[selected[0][0]], 0.3, np.zeros_like(frames[selected[0][0]]), 0.7, 0)
    for x in (0.2 * w, 0.8 * w):
        cv2.line(coverage, (int(x), 0), (int(x), h), (80, 80, 80), 2)
    for y in (0.2 * h, 0.8 * h):
        cv2.line(coverage, (0, int(y)), (w, int(y)), (80, 80, 80), 2)
    for i, d in enumerate(selected):
        color = tuple(int(c) for c in cv2.applyColorMap(np.uint8([[i * 255 // len(selected)]]), cv2.COLORMAP_JET)[0, 0])
        for x, y in d[1]:
            cv2.circle(coverage, (int(x), int(y)), 5, color, -1)
    cv2.imwrite(str(debug_dir / "coverage.png"), coverage)

    # Before/after undistortion on the frame whose board is closest to an image corner,
    # where distortion is strongest.
    far = max(selected, key=lambda d: np.linalg.norm(d[1].mean(axis=0) - (w / 2, h / 2)))
    before = frames[far[0]]
    after = cv2.undistort(before, K, dist)
    side = np.hstack([before, np.full((h, 20, 3), 255, np.uint8), after])
    cv2.putText(side, "original", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)
    cv2.putText(side, "undistorted", (w + 50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)
    cv2.imwrite(str(debug_dir / f"undistort_frame{far[0]:05d}.png"), side)

    # Summary.
    corners_hit = corner_coverage(all_points, w, h)
    missing = [name for name, n in corners_hit.items() if n == 0]
    k = dist.ravel()
    print("\n=== Calibration summary ===")
    print(f"image size : {w}x{h}")
    print(f"fx, fy     : {K[0, 0]:.2f}, {K[1, 1]:.2f}")
    print(f"cx, cy     : {K[0, 2]:.2f}, {K[1, 2]:.2f}")
    print(f"distortion : k1={k[0]:.5f} k2={k[1]:.5f} p1={k[2]:.5f} p2={k[3]:.5f} k3={k[4]:.5f}")
    print(f"RMS error  : {rms:.3f} px", end="")
    if rms > 1.0:
        print("  WARNING: above 1 px, calibration is unreliable")
    elif rms > 0.5:
        print("  (above the 0.5 px target)")
    else:
        print("  (good)")
    print(f"frames used: {len(selected)}")
    print("corner coverage (points in outer 20% corner regions): "
          + ", ".join(f"{name}={n}" for name, n in corners_hit.items()))
    if missing:
        print(f"  WARNING: no corners detected near the {', '.join(missing)} image corner(s); "
              "distortion there is extrapolated. Consider re-recording with the board pushed into those corners.")
    else:
        print("  all four image corners covered")
    print(f"\nSaved {out_path} and debug images in {debug_dir}/")


if __name__ == "__main__":
    main()
