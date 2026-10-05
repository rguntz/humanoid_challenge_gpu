"""Run the whole video -> simulation pipeline on every episode clip, one after the other.

For each clip in videos/task/dataset/ (only clips whose export is finished, as
recorded in the *_episodes.json written by the episode cutter), run:

  1. tag_pose/estimate_tag_pose.py   camera pose from the tag (first still second)
  2. hand_tracking/track_hands.py     3D hand keypoints (WiLoR), hand scale from the rests
  3. gripper/virtual_gripper.py      virtual gripper + single grasp / release
  4. gripper/depth_correct.py        depth anchors at the rests, the grasp, the release
  5. sim/build_scene.py              table, Panda, tag, phone camera, bottle, container
  6. sim/replay_hand.py --no-video   IK joint trajectory
  7. sim/rollout_physics.py          physics rollout, success check, recorded demo + video

then pipeline/make_robomimic.py merges the successful episodes into one robomimic
HDF5 dataset.

Outputs, in videos/task/dataset_sim/:
  videos/<clip>.mp4        the physics video (real video + sim overlay | overview)
  episodes/<clip>/         every intermediate file, demo.hdf5 and a log.txt of all steps
  summary.csv              one row per episode: status, failing step, key numbers
  robomimic/pick_place_bottle.hdf5   successful episodes only (+ _episodes.csv: kept / excluded)

On a headless Linux machine (no display), MuJoCo renders with EGL (MUJOCO_GL=egl).
With --gpus, each episode's steps are pinned to one GPU (CUDA_VISIBLE_DEVICES and
MUJOCO_EGL_DEVICE_ID); only the main process writes summary.csv and the dataset.

A failing episode is logged and skipped; the batch goes on. Episodes that already
have a physics result are skipped unless --force, so the batch can be resumed.

Usage:
    uv run pipeline/run_dataset.py                 # all episodes
    uv run pipeline/run_dataset.py --only 1 2 3    # some episodes (by number)
    uv run pipeline/run_dataset.py --force         # redo everything
    uv run pipeline/run_dataset.py --only 1 --check-videos   # one episode with every check video
    uv run pipeline/run_dataset.py --gpus 0,2      # 2 episodes in parallel, on GPUs 0 and 2
    uv run pipeline/run_dataset.py --gpus 0,0,2,2  # 4 in parallel, 2 per GPU
"""
import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).parent.parent
DATASET = ROOT / "videos" / "task" / "dataset"
OUT = ROOT / "videos" / "task" / "dataset_sim"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", type=int, nargs="*", help="episode numbers to process (default: all)")
    p.add_argument("--force", action="store_true", help="redo episodes that already have a result")
    p.add_argument("--tag-size", type=float, default=150.0, help="measured tag side, mm (default 150)")
    p.add_argument("--no-video", action="store_true", help="skip the physics check videos (faster)")
    p.add_argument("--gpus", default=None,
                   help="run several episodes in parallel, one per listed GPU slot, e.g. 0,2 (2 at a time) "
                        "or 0,0,2,2 (2 per GPU). GPU numbers as in nvidia-smi. Default: one at a time")
    p.add_argument("--check-videos", action="store_true",
                   help="also write the intermediate check videos (tag axes, hand keypoints, virtual gripper) "
                        "in episodes/<clip>/")
    return p.parse_args()


def ready_clips():
    """Clips whose export finished (listed as exported with a file on disk)."""
    clips = []
    for meta in sorted(DATASET.glob("*_episodes.json")):
        for ep in json.loads(meta.read_text())["episodes"]:
            exp = ep.get("exported")
            if exp and (DATASET / exp["file"]).exists() \
                    and exp["start_frame"] == ep["start_frame"] and exp["end_frame"] == ep["end_frame"]:
                clips.append((ep["id"], DATASET / exp["file"], ep.get("note", "")))
    return sorted(clips, key=lambda c: c[1].name)


def run_episode(clip, args, gpu=None):
    """Run every step on one clip. gpu: GPU index (as in nvidia-smi) the steps are
    pinned to, for both PyTorch (CUDA) and MuJoCo rendering (EGL); None = unchanged."""
    stem = clip.stem
    d = OUT / "episodes" / stem
    d.mkdir(parents=True, exist_ok=True)
    video_out = OUT / "videos" / f"{stem}.mp4"
    py = sys.executable
    skip = [] if args.check_videos else ["--no-video"]  # intermediate check videos
    steps = [
        ("tag_pose", [py, ROOT / "tag_pose/estimate_tag_pose.py", clip, "--tag-size", str(args.tag_size),
                      "--out", d / "tag_pose.json", "--out-video", d / "tag_axes.mp4"] + skip),
        ("hands", [py, ROOT / "hand_tracking/track_hands.py", clip, "--tag-pose", d / "tag_pose.json",
                   "--out", d / "hands.json", "--out-video", d / "hands.mp4"] + skip),
        ("gripper", [py, ROOT / "gripper/virtual_gripper.py", d / "hands.json", "--out", d / "gripper.json"]
         + skip),
        ("depth", [py, ROOT / "gripper/depth_correct.py", d / "gripper.json",
                   "--out", d / "gripper_corrected.json"]),
        ("scene", [py, ROOT / "sim/build_scene.py", clip, "--tag-pose", d / "tag_pose.json",
                   "--gripper", d / "gripper_corrected.json", "--out-dir", d]),
        ("ik", [py, ROOT / "sim/replay_hand.py", clip, "--scene", d / "scene.xml",
                "--gripper", d / "gripper_corrected.json", "--out", d / "robot.mp4", "--no-video"]),
        ("physics", [py, ROOT / "sim/rollout_physics.py", clip, "--scene", d / "scene.xml",
                     "--robot", d / "robot.json", "--out", video_out, "--out-json", d / "physics.json",
                     "--record", d / "demo.hdf5"] + (["--no-video"] if args.no_video else [])),
    ]
    env = os.environ.copy()
    if sys.platform.startswith("linux") and not env.get("DISPLAY"):
        env.setdefault("MUJOCO_GL", "egl")  # headless rendering
    if gpu is not None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # CUDA numbering = nvidia-smi numbering
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)   # hand tracking sees only this GPU
        env["MUJOCO_EGL_DEVICE_ID"] = str(gpu)   # MuJoCo renders on the same GPU
    (d / "demo.hdf5").unlink(missing_ok=True)  # never keep a demo from an older run
    log = []
    for name, cmd in steps:
        t0 = time.time()
        r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=ROOT / "sim", env=env)
        out = (r.stdout + r.stderr).replace("\r", "\n")
        log.append(f"===== {name} ({time.time() - t0:.0f} s, exit {r.returncode}) =====\n{out}")
        (d / "log.txt").write_text("\n".join(log))
        if r.returncode != 0:
            lines = [l for l in out.strip().splitlines() if l.strip()]
            return {"status": "failed", "failed_step": name, "error": lines[-1][:200] if lines else ""}
    return {"status": "ok", "failed_step": "", "error": ""}


def summarize(clip, result):
    """Key numbers from the intermediate files, for the summary table."""
    d = OUT / "episodes" / clip.stem
    row = {"clip": clip.name, **result}

    def load(name):
        p = d / name
        return json.loads(p.read_text()) if p.exists() else None

    tag = load("tag_pose.json")
    if tag:
        row["tag_reproj_px"] = round(tag["reprojection_error_px"], 2)
        row["camera_height_m"] = round(tag["camera_position_in_tag_m"][2], 3)
    hands = load("hands.json")
    if hands:
        row["hand_frames"] = f"{sum(1 for f in hands['frames'] if f['hands'])}/{len(hands['frames'])}"
        row["hand_scale"] = round(hands["hand_scale"], 3)
    corr = load("gripper_corrected.json")
    if corr:
        ev = {e["event"]: e["frame"] for e in corr["events"]}
        row["grasp_frame"], row["release_frame"] = ev.get("close"), ev.get("open")
        for a in corr["depth_correction"]["anchors"]:
            if a["name"] in ("grasp", "release"):
                row[f"depth_scale_{a['name']}"] = a["scale"]
    robot = load("robot.json")
    if robot:
        row["ik_err_max_mm"] = max(robot["ik_position_error_mm"])
    phys = load("physics.json")
    if phys:
        row["lifted_cm"] = round(100 * phys["bottle_lifted_m"], 1)
        row["inside_container"] = phys["inside_container"]
        row["bottle_tilt_deg"] = phys["bottle_tilt_deg"]
        row["arm_lag_max_deg"] = phys.get("arm_tracking_error_max_deg")
        row["success"] = bool(phys.get("success"))
    return row


def main():
    args = parse_args()
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    clips = ready_clips()
    if args.only:
        clips = [c for c in clips if c[0] in args.only]
    print(f"{len(clips)} episode clips to process -> {OUT}")

    summary_path = OUT / "summary.csv"
    rows = {}
    if summary_path.exists():
        with open(summary_path) as f:
            rows = {r["clip"]: r for r in csv.DictReader(f)}

    todo = []
    for ep_id, clip, note in clips:
        done = (OUT / "episodes" / clip.stem / "physics.json").exists() and \
               (OUT / "episodes" / clip.stem / "demo.hdf5").exists()
        if done and not args.force:
            print(f"{clip.name}: already done, skipped")
            rows.setdefault(clip.name, summarize(clip, {"status": "ok", "failed_step": "", "error": ""}))
        else:
            todo.append(clip)

    # One worker per GPU slot (--gpus 0,2 -> 2 episodes at a time; 0,0,2,2 -> 4).
    # Each worker takes a free slot, runs one episode pinned to that GPU, gives it back.
    slots = queue.Queue()
    gpus = [g.strip() for g in args.gpus.split(",")] if args.gpus else [None]
    for g in gpus:
        slots.put(g)
    if args.gpus:
        print(f"{len(todo)} episodes on GPUs {args.gpus} ({len(gpus)} at a time)")

    def work(clip):
        gpu = slots.get()
        try:
            t0 = time.time()
            return clip, gpu, run_episode(clip, args, gpu), time.time() - t0
        finally:
            slots.put(gpu)

    # Progress bar over the episodes (with time remaining); one result line per
    # finished episode is printed above it.
    progress = tqdm(total=len(todo), unit="episode", dynamic_ncols=True, desc="episodes")
    n_success = 0
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(work, clip) for clip in todo]
        for n_done, fut in enumerate(as_completed(futures), 1):
            clip, gpu, result, elapsed = fut.result()
            rows[clip.name] = summarize(clip, result)
            r = rows[clip.name]
            where = f", GPU {gpu}" if gpu is not None else ""
            head = f"[{n_done}/{len(todo)}] {clip.name}"
            if result["status"] == "ok":
                tqdm.write(f"{head}: ok ({elapsed:.0f} s{where}): lifted {r.get('lifted_cm')} cm, "
                           f"{'inside' if r.get('inside_container') else 'OUTSIDE'}, "
                           f"tilt {r.get('bottle_tilt_deg')} deg"
                           + ("  -> SUCCESS" if r.get("success") else "  -> not in the box"))
            else:
                tqdm.write(f"{head}: FAILED at {result['failed_step']} ({elapsed:.0f} s{where}): {result['error']}")
            n_success += bool(r.get("success"))
            progress.set_postfix(success=f"{n_success}/{n_done}")
            progress.update(1)

            # Rewrite the summary after every episode (only this main thread writes it),
            # so it is up to date if the batch is interrupted.
            fields = []
            for row in rows.values():
                fields += [k for k in row if k not in fields]
            with open(summary_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for name in sorted(rows):
                    w.writerow(rows[name])

    progress.close()

    ok = [r for r in rows.values() if r.get("status") == "ok"]
    succ = [r for r in ok if str(r.get("success")) == "True"]
    print(f"\n{len(rows)} episodes: {len(ok)} ran through the pipeline, {len(succ)} successful in physics "
          f"(bottle in the box).\nSummary: {summary_path}\nVideos: {OUT / 'videos'}")
    if succ:
        print("\nBuilding the robomimic dataset from the successful episodes:")
        subprocess.run([sys.executable, str(ROOT / "pipeline/make_robomimic.py")], check=False)


if __name__ == "__main__":
    main()
