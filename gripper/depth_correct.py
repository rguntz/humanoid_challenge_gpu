"""Correct the virtual gripper's depth with known heights at a few anchor moments.

Monocular hand tracking gets the direction from the camera right (the overlay
matches the video) but not the distance along that ray: it depends on the hand's
apparent size, which WiLoR misjudges when the hand changes shape (e.g. curled
around a cap). Moving a point along its camera ray is a scaling about the camera
centre, which leaves its image position unchanged. So per frame we scale the
gripper position about the camera by a factor s(t), chosen at anchor moments
where the true height is known:

  start / end rest : hand flat on the table, already calibrated by the hand
                     scale in track_hands.py -> s = 1
  grasp (close)    : fingertips on the bottle at --grasp-height above the table
  release (open)   : same height above the container floor (--place-floor), plus
                     a small --place-clearance: the tracked height is only good to
                     about 1 cm, and a stiff robot pressing the object into the
                     floor squeezes it out of the fingers, so it is released just
                     above the floor and drops the last bit

and interpolated linearly in time in between (constant before the first and
after the last anchor). The opening is scaled too (it is a 3D distance);
orientation and the open/closed state are unchanged.

Usage:
    uv run gripper/depth_correct.py gripper/PXL_20261005_120635999_gripper.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
HERE = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gripper_json", help="output of virtual_gripper.py")
    p.add_argument("--grasp-height", type=float, default=0.0675,
                   help="fingertip height on the object when grasped, m (default 0.0675: middle of the cap "
                        "of the 7.5 cm bottle)")
    p.add_argument("--place-clearance", type=float, default=0.01,
                   help="release this far above the floor, m (default 0.01)")
    p.add_argument("--place-floor", type=float, default=0.003,
                   help="height of the container floor the object is released onto, m (default 0.003)")
    p.add_argument("--n-frames", type=int, default=6,
                   help="frames averaged at the grasp and release anchors (default 6)")
    p.add_argument("--out", default=None, help="default gripper/<video>_gripper_corrected.json")
    return p.parse_args()


def scale_to_height(p, cam, z):
    """Factor s such that cam + s (p - cam) is at height z."""
    return (z - cam[2]) / (p[2] - cam[2])


def main():
    args = parse_args()
    g = json.loads(Path(args.gripper_json).read_text())
    hands = json.loads(Path(g["hands_json"]).read_text())
    tag_pose = json.loads(Path(hands["tag_pose"]).read_text())
    cam = np.array(tag_pose["T_tag_cam"])[:3, 3]
    stem = Path(g["video"]).stem
    out_path = Path(args.out) if args.out else HERE / f"{stem}_gripper_corrected.json"

    frames = g["frames"]
    n = len(frames)
    P = np.array([f["position"] for f in frames])
    events = {e["event"]: e["frame"] for e in g["events"]}
    close, release = events["close"], events.get("open")

    # Anchors (frame, scale).
    anchors = []
    rest = set(hands.get("rest_frames", []))
    if 0 in rest:  # end of the start rest
        last = 0
        while last + 1 in rest:
            last += 1
        anchors.append((last, 1.0, "start rest"))
    p_close = np.median(P[close:close + args.n_frames], axis=0)
    anchors.append((close, scale_to_height(p_close, cam, args.grasp_height), "grasp"))
    if release is not None:
        p_release = np.median(P[max(close, release - args.n_frames):release + 1], axis=0)
        place_height = args.place_floor + args.place_clearance + args.grasp_height
        anchors.append((release, scale_to_height(p_release, cam, place_height),
                        "release"))
    if n - 1 in rest:  # start of the end rest
        first = n - 1
        while first - 1 in rest:
            first -= 1
        if first > anchors[-1][0]:
            anchors.append((first, 1.0, "end rest"))

    a_frames = [a[0] for a in anchors]
    a_scales = [a[1] for a in anchors]
    s = np.interp(np.arange(n), a_frames, a_scales)

    corrected = []
    for f, si in zip(frames, s):
        f = dict(f)
        p = np.array(f["position"])
        f["position"] = (cam + si * (p - cam)).round(5).tolist()
        f["opening"] = round(f["opening"] * si, 5)
        f["depth_scale"] = round(float(si), 4)
        corrected.append(f)
    P_new = np.array([f["position"] for f in corrected])

    print("depth anchors (scale about the camera; <1 = closer to the camera):")
    for (fr, sc, name) in anchors:
        print(f"  {name:10s} frame {fr:4d}: scale {sc:.3f}  height {100 * P[fr, 2]:5.1f} -> "
              f"{100 * P_new[fr, 2]:5.1f} cm, moved {100 * np.linalg.norm(P_new[fr] - P[fr]):.1f} cm")

    result = dict(g)
    result["frames"] = corrected
    result["depth_correction"] = {
        "source": str(args.gripper_json),
        "camera_position": cam.round(5).tolist(),
        "grasp_height": args.grasp_height,
        "place_floor": args.place_floor,
        "place_clearance": args.place_clearance,
        "anchors": [{"frame": fr, "scale": round(sc, 4), "name": name} for fr, sc, name in anchors],
    }
    out_path.write_text(json.dumps(result))
    print(f"Saved {out_path}")

    # Plot: height before / after and the scale factor.
    t = np.arange(n) / g["fps"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), dpi=150, sharex=True)
    axes[0].plot(t, 100 * P[:, 2], color="#2a78d6", lw=1.2, alpha=0.5, label="tracked")
    axes[0].plot(t, 100 * P_new[:, 2], color="#2a78d6", lw=2, label="depth-corrected")
    axes[0].axhline(100 * args.grasp_height, color="#8f8e86", lw=1, ls="--")
    axes[0].text(t[-1], 100 * args.grasp_height, " grasp height", va="center", fontsize=8, color="#55544e")
    axes[0].set_ylabel("gripper height (cm)", color="#55544e")
    axes[0].set_title("Depth correction: anchors at the rests, the grasp and the release", loc="left",
                      fontsize=10, color="#1f1e1b")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    axes[1].plot(t, s, color="#eb6834", lw=2)
    for fr, sc, name in anchors:
        axes[1].plot(t[fr], sc, "o", color="#eb6834", ms=6)
        axes[1].annotate(name, (t[fr], sc), xytext=(0, 8), textcoords="offset points", ha="center",
                         fontsize=8, color="#55544e")
    axes[1].set_ylabel("scale about camera", color="#55544e")
    axes[1].set_xlabel("time (s)", color="#55544e")
    for ax in axes:
        ax.grid(axis="y", color="#e2e1da", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors="#55544e", labelsize=8)
    fig.tight_layout()
    plot_path = out_path.with_suffix(".png")
    fig.savefig(plot_path)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
