"""Check hand heights above the table (tag frame z) from track_hands.py output.

The tag frame has its origin at the tag centre, x/y in the table plane and z up
out of the printed face, so z of a keypoint is its height above the table. While
the hand rests flat on the table (the rest periods at the start and end of the
video, as detected by track_hands.py),
fingertips and palm should be at about 0-2 cm and the wrist a few cm higher.

Keypoints are stored in metres (what MuJoCo uses); heights are shown in cm.

Usage:
    uv run hand_tracking/check_table_height.py hand_tracking/PXL_20261005_120635999_hands.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GROUPS = {  # name: (joint indices, colour, expected resting range in cm)
    "wrist": ([0], "#2a78d6", (0.0, 6.0)),
    "palm (knuckles)": ([5, 9, 13, 17], "#eb6834", (-1.0, 3.0)),
    "thumb tip": ([4], "#1baf7a", (-1.0, 3.0)),
    "index tip": ([8], "#e87ba4", (-1.0, 3.0)),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hands_json", help="output of track_hands.py")
    p.add_argument("--out", default=None, help="plot path (default <json stem>_heights.png)")
    args = p.parse_args()

    data = json.loads(Path(args.hands_json).read_text())
    if not data.get("tag_pose"):
        raise SystemExit("No tag-frame keypoints in this file: run track_hands.py with a tag pose.")
    out = Path(args.out) if args.out else Path(args.hands_json).with_name(Path(args.hands_json).stem + "_heights.png")

    # First hand per frame (track_hands.py keeps one per hand); NaN where no hand.
    t = np.array([f["time_s"] for f in data["frames"]])
    z = np.full((len(t), 21), np.nan)
    for i, f in enumerate(data["frames"]):
        if f["hands"]:
            z[i] = np.array(f["hands"][0]["keypoints_3d_tag"])[:, 2] * 100  # cm

    heights = {name: np.nanmean(z[:, idx], axis=1) for name, (idx, _, _) in GROUPS.items()}
    # Rest periods found by track_hands.py (hand still at the start / end of the video).
    rest = np.zeros(len(t), bool)
    rest[data["rest_frames"]] = True
    windows = {}
    if rest[0]:
        windows["start"] = rest & (np.cumsum(~rest) == 0)
    if rest[-1]:
        windows["end"] = rest & (np.cumsum(~rest[::-1])[::-1] == 0)
    if not windows:
        raise SystemExit("No rest period found at the start or end of the video.")

    print(f"Hand height above the table (cm), hand scale {data.get('hand_scale', 1.0):.3f}")
    print(f"{'':18s}" + "".join(f"{w + ' rest':>14s}" for w in windows) + "   expected")
    all_ok = True
    for name, (_, _, (lo, hi)) in GROUPS.items():
        row = f"{name:18s}"
        for mask in windows.values():
            m = np.nanmean(heights[name][mask])
            ok = lo <= m <= hi
            all_ok &= ok
            row += f"{m:10.1f} {'ok ' if ok else 'BAD'} "
        print(row + f"  {lo:g} to {hi:g}")
    print("\nResting hand is on the table." if all_ok else
          "\nWARNING: resting hand is off the table: check the hand scale, intrinsics or tag size.")
    print("Note: one hand scale is calibrated on all rest frames, so the check is that the start "
          "and end rests agree with each other and with the expected range.")

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)
    ax.axhspan(0, 2, color="#e9e8e2", zorder=0)
    ax.text(t[0] + 0.05, 1.0, "on the table (0-2 cm)", va="center", fontsize=8, color="#55544e")
    for mask in windows.values():
        ax.axvspan(t[mask][0], t[mask][-1], color="#f4f3ee", zorder=0)
    ax.axhline(0, color="#8f8e86", lw=1)
    for name, (_, color, _) in GROUPS.items():
        ax.plot(t, heights[name], color=color, lw=2, label=name)
        last = np.where(~np.isnan(heights[name]))[0][-1]
        ax.annotate(name, (t[last], heights[name][last]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color="#3d3c37")
    ax.set_xlabel("time (s)", color="#55544e")
    ax.set_ylabel("height above table (cm)", color="#55544e")
    ax.set_title("Hand height above the table (tag frame z); shaded columns = rest windows", loc="left",
                 fontsize=10, color="#1f1e1b")
    ax.grid(axis="y", color="#e2e1da", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#55544e", labelsize=8)
    ax.set_xlim(t[0], t[-1] + 0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
