"""Turn tracked hand keypoints into a virtual parallel gripper, smoothed over time.

Per frame, in the tag (table) frame, in metres:
  position : midpoint of the thumb tip and index tip
  opening  : thumb tip - index tip distance
  axes     : y = closing direction (thumb tip -> index tip)
             z = approach direction (from the midpoint of the thumb and index
                 knuckles towards the gripper position), made orthogonal to y
             x = y cross z (right-handed)

Smoothing (offline, zero lag): a short median filter removes single-frame depth
spikes, then a Savitzky-Golay filter smooths position, opening and the y / approach
vectors; the axes are re-orthonormalised after smoothing.

Open / closed state, assuming a pick-and-place episode with exactly one grasp:
  grasp : the frame with the smallest opening is inside the grasp; the grasp
          segment is the run around it where the opening stays below halfway
          between the typical closed (10th percentile) and open (90th percentile)
          openings.
  close : first frame of that segment where the opening is below --close-below
          (default 15% of the way from closed to open: the fingers have settled on
          the object, not halfway through closing).
  open  : the release is in the final rise of the opening at the end of the
          segment: find where that rise starts, then the first frame more than
          --open-margin above it. The grip drifts by several mm while carrying, so
          this only looks at the final rise and never mid-carry.

Usage:
    uv run gripper/virtual_gripper.py hand_tracking/PXL_20261005_120635999_hands.json
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt, savgol_filter
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).parent.parent
HERE = Path(__file__).parent

THUMB_MCP, THUMB_TIP, INDEX_MCP, INDEX_TIP = 2, 4, 5, 8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hands_json", help="output of hand_tracking/track_hands.py")
    p.add_argument("--window", type=float, default=0.3, help="Savitzky-Golay window, seconds (default 0.3)")
    p.add_argument("--median", type=int, default=5, help="median filter length, frames, odd (default 5)")
    p.add_argument("--close-below", type=float, default=None,
                   help="opening (mm) below which the gripper closes (default: automatic)")
    p.add_argument("--open-margin", type=float, default=12.0,
                   help="release when the opening has risen this many mm from the start of its final "
                        "rise (default 12)")
    p.add_argument("--out", default=None, help="output JSON (default gripper/<video>_gripper.json)")
    p.add_argument("--no-video", action="store_true", help="skip the overlay video")
    return p.parse_args()


def gripper_frame(y_dir, approach):
    """Rotation matrix with columns x, y, z from a closing direction and an approach vector."""
    y = y_dir / np.linalg.norm(y_dir, axis=-1, keepdims=True)
    z = approach - np.sum(approach * y, axis=-1, keepdims=True) * y
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=-1)


def interpolate_missing(values, valid):
    """Fill frames without a hand by linear interpolation (per column)."""
    out = values.copy()
    idx = np.arange(len(values))
    for c in range(values.shape[1]):
        out[~valid, c] = np.interp(idx[~valid], idx[valid], values[valid, c])
    return out


def find_grasp(opening, close_below, mid, open_margin):
    """Frames (close, release) of the single grasp; closed for close <= frame < release."""
    g = int(np.argmin(opening))
    if opening[g] >= close_below:
        raise SystemExit("The opening never drops below the close threshold: no grasp found.")
    # Grasp segment: run around the tightest grip where the opening stays below mid.
    start, end = g, g
    while start > 0 and opening[start - 1] < mid:
        start -= 1
    while end < len(opening) - 1 and opening[end + 1] < mid:
        end += 1
    close = start + int(np.argmax(opening[start:end + 1] < close_below))
    # Final rise: walk back from the end of the segment while the opening keeps
    # decreasing (1 mm tolerance for wiggles), then release once it has risen
    # open_margin above that point.
    rise = end
    while rise > close and opening[rise - 1] < opening[rise] + 0.001:
        rise -= 1
    above = np.flatnonzero(opening[rise:] > opening[rise] + open_margin)
    release = rise + int(above[0]) if len(above) else len(opening)
    return close, release


def smooth(values, median, window):
    values = np.stack([medfilt(values[:, c], median) for c in range(values.shape[1])], axis=1)
    return savgol_filter(values, window, polyorder=2, axis=0, mode="interp")


def main():
    args = parse_args()
    data = json.loads(Path(args.hands_json).read_text())
    if not data.get("tag_pose"):
        raise SystemExit("No tag-frame keypoints in this file: run track_hands.py with a tag pose.")
    stem = Path(data["video"]).stem
    out_json = Path(args.out) if args.out else HERE / f"{stem}_gripper.json"
    fps = data["fps"]
    n = len(data["frames"])

    # Raw keypoints in the tag frame (first hand per frame).
    kp = np.full((n, 21, 3), np.nan)
    for i, f in enumerate(data["frames"]):
        if f["hands"]:
            kp[i] = f["hands"][0]["keypoints_3d_tag"]
    valid = ~np.isnan(kp[:, 0, 0])
    if valid.sum() < 2:
        raise SystemExit("Not enough frames with a hand.")

    thumb, index = kp[:, THUMB_TIP], kp[:, INDEX_TIP]
    knuckles = (kp[:, THUMB_MCP] + kp[:, INDEX_MCP]) / 2
    pos_raw = (thumb + index) / 2
    opening_raw = np.linalg.norm(index - thumb, axis=1)
    ydir_raw = index - thumb
    approach_raw = pos_raw - knuckles
    R_raw = gripper_frame(ydir_raw, approach_raw)

    # Smooth position, opening, and the two direction vectors, then rebuild the frame.
    window = max(5, int(round(args.window * fps)) | 1)  # odd, at least 5 frames
    signals = np.concatenate([pos_raw, opening_raw[:, None], ydir_raw, approach_raw], axis=1)
    signals = smooth(interpolate_missing(signals, valid), args.median, window)
    pos = signals[:, 0:3]
    opening = signals[:, 3]
    R = gripper_frame(signals[:, 4:7], signals[:, 7:10])
    quat_xyzw = Rotation.from_matrix(R).as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]  # MuJoCo order

    # Open / closed state (metres).
    w_closed, w_open = np.percentile(opening, [10, 90])
    if args.close_below is not None:
        close_below = args.close_below / 1000
    else:
        close_below = w_closed + 0.15 * (w_open - w_closed)
        if w_open - w_closed < 0.02:
            print("WARNING: opening barely changes in this video; the automatic threshold may be meaningless.")
    open_margin = args.open_margin / 1000
    close_frame, release_frame = find_grasp(opening, close_below, (w_closed + w_open) / 2, open_margin)
    closed = np.zeros(n, bool)
    closed[close_frame:release_frame] = True
    events = [{"frame": close_frame, "time_s": round(close_frame / fps, 3), "event": "close"}]
    if release_frame < n:
        events.append({"frame": release_frame, "time_s": round(release_frame / fps, 3), "event": "open"})
    else:
        print("WARNING: no release found; the gripper stays closed until the end.")

    frames_out = []
    for i in range(n):
        frames_out.append({
            "frame": i,
            "time_s": round(i / fps, 4),
            "hand_detected": bool(valid[i]),
            "position": pos[i].round(5).tolist(),
            "rotation": R[i].round(5).tolist(),
            "quat_wxyz": quat_wxyz[i].round(5).tolist(),
            "opening": round(float(opening[i]), 5),
            "closed": bool(closed[i]),
            "raw": None if not valid[i] else {
                "position": pos_raw[i].round(5).tolist(),
                "rotation": R_raw[i].round(5).tolist(),
                "opening": round(float(opening_raw[i]), 5),
            },
        })
    result = {
        "video": data["video"],
        "hands_json": str(args.hands_json),
        "fps": fps,
        "frame": "tag frame: origin at tag centre, x/y on the table, z up; metres",
        "axes": "rotation columns = gripper x, y (closing: thumb -> index), z (approach)",
        "smoothing": {"median_frames": args.median, "savgol_frames": window, "savgol_polyorder": 2},
        "state_thresholds": {"close_below": round(close_below, 4), "open_margin": open_margin},
        "events": events,
        "frames": frames_out,
    }
    out_json.write_text(json.dumps(result))

    # Smoothing effect: frame-to-frame jitter before / after.
    def jitter(x):
        return np.median(np.linalg.norm(np.diff(x[valid], axis=0).reshape(valid.sum() - 1, -1), axis=1))
    ang_raw = np.degrees(np.arccos(np.clip((np.trace(np.swapaxes(R_raw[valid][:-1], 1, 2) @ R_raw[valid][1:],
                                                      axis1=1, axis2=2) - 1) / 2, -1, 1)))
    ang = np.degrees(np.arccos(np.clip((np.trace(np.swapaxes(R[:-1], 1, 2) @ R[1:], axis1=1, axis2=2) - 1) / 2,
                                       -1, 1)))
    print(f"{n} frames, hand in {valid.sum()}; smoothing: median {args.median} + Savitzky-Golay {window} frames")
    print(f"median frame-to-frame change   raw -> smoothed")
    print(f"  position (mm)               {1000 * jitter(pos_raw):6.1f} -> {1000 * jitter(pos):5.1f}")
    print(f"  opening (mm)                {1000 * jitter(opening_raw[:, None]):6.1f} -> "
          f"{1000 * jitter(opening[:, None]):5.1f}")
    print(f"  orientation (deg)           {np.median(ang_raw):6.1f} -> {np.median(ang):5.1f}")
    print(f"opening range: {1000 * opening.min():.0f} - {1000 * opening.max():.0f} mm")
    print(f"single grasp: close below {1000 * close_below:.0f} mm, release {1000 * open_margin:.0f} mm "
          "into the final rise")
    for e in events:
        print(f"  {e['event']:5s} at frame {e['frame']:4d} ({e['time_s']:.2f} s)")
    print(f"closed in {closed.sum()} of {n} frames")

    plot_path = out_json.with_suffix(".png")
    plot(np.arange(n) / fps, valid, pos_raw, pos, opening_raw, opening, closed, close_below, plot_path)
    print(f"Saved {out_json}\nSaved {plot_path}")

    if not args.no_video:
        video_path = out_json.with_suffix(".mp4")
        write_video(data, pos, R, opening, closed, thumb, index, valid, video_path)
        print(f"Saved {video_path}")


def plot(t, valid, pos_raw, pos, opening_raw, opening, closed, close_below, path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), dpi=150, sharex=True)
    # Shade the closed periods on both panels.
    edges = np.flatnonzero(np.diff(np.r_[0, closed.astype(int), 0]))
    for start, stop in zip(edges[::2], edges[1::2]):
        for ax in axes:
            ax.axvspan(t[start], t[stop - 1], color="#ecebe5", zorder=0)
        axes[1].text((t[start] + t[stop - 1]) / 2, 1000 * opening.max(), "closed", ha="center",
                     fontsize=8, color="#55544e")
    axes[1].axhline(1000 * close_below, color="#8f8e86", lw=1, ls="--")
    axes[1].text(t[-1], 1000 * close_below, " close below", va="center", fontsize=8, color="#55544e")
    series = [("x", "#2a78d6"), ("y", "#eb6834"), ("z (height)", "#1baf7a")]
    for c, (name, color) in enumerate(series):
        axes[0].plot(t[valid], 100 * pos_raw[valid, c], color=color, lw=1, alpha=0.35)
        axes[0].plot(t, 100 * pos[:, c], color=color, lw=2, label=name)
    axes[0].set_ylabel("position in tag frame (cm)", color="#55544e")
    axes[0].set_title("Virtual gripper: thin = raw, thick = smoothed, shaded = closed", loc="left", fontsize=10,
                      color="#1f1e1b")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8, ncol=3)
    axes[1].plot(t[valid], 1000 * opening_raw[valid], color="#e87ba4", lw=1, alpha=0.35)
    axes[1].plot(t, 1000 * opening, color="#e87ba4", lw=2)
    axes[1].set_ylabel("opening (mm)", color="#55544e")
    axes[1].set_xlabel("time (s)", color="#55544e")
    for ax in axes:
        ax.grid(axis="y", color="#e2e1da", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors="#55544e", labelsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_video(data, pos, R, opening, closed, thumb, index, valid, path):
    """Overlay the smoothed gripper frame on the video (tag frame -> camera -> image)."""
    calib = json.loads(Path(data["intrinsics"]).read_text())
    K, dist = np.array(calib["camera_matrix"]), np.array(calib["dist_coeffs"])
    T_cam_tag = np.array(json.loads(Path(data["tag_pose"]).read_text())["T_cam_tag"])
    rvec, tvec = cv2.Rodrigues(T_cam_tag[:3, :3])[0], T_cam_tag[:3, 3]

    def project(points):
        img, _ = cv2.projectPoints(np.asarray(points, float).reshape(-1, 3), rvec, tvec, K, dist)
        return img.reshape(-1, 2).astype(int)

    cap = cv2.VideoCapture(data["video"])
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), data["fps"],
                             (data["image_width"], data["image_height"]))
    axis_len = 0.05
    for i in range(len(pos)):
        ok, frame = cap.read()
        if not ok:
            break
        state_color = (0, 0, 230) if closed[i] else (0, 170, 0)  # BGR: red closed, green open
        if valid[i]:
            tips = project([thumb[i], index[i]])
            cv2.line(frame, tuple(tips[0]), tuple(tips[1]), state_color, 3, cv2.LINE_AA)
            for p in tips:
                cv2.circle(frame, tuple(p), 7, (255, 255, 255), -1, cv2.LINE_AA)
        o, *ends = project([pos[i]] + [pos[i] + axis_len * R[i][:, k] for k in range(3)])
        for end, color in zip(ends, [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):  # x red, y green, z blue
            cv2.line(frame, tuple(o), tuple(end), color, 4, cv2.LINE_AA)
        cv2.circle(frame, tuple(o), 10, state_color, -1, cv2.LINE_AA)
        cv2.putText(frame, f"frame {i}  opening {1000 * opening[i]:.0f} mm  height {100 * pos[i][2]:.1f} cm",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        label = "CLOSED" if closed[i] else "OPEN"
        cv2.rectangle(frame, (20, 75), (330, 165), state_color, -1)
        cv2.putText(frame, label, (35, 145), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255, 255, 255), 6, cv2.LINE_AA)
        writer.write(frame)
    cap.release()
    writer.release()


if __name__ == "__main__":
    main()
