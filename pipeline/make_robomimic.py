"""Merge the successful simulated episodes into one robomimic HDF5 dataset.

Reads videos/task/dataset_sim/episodes/<clip>/demo.hdf5 (one recorded physics
rollout per episode, written by sim/rollout_physics.py --record) and keeps only the
episodes whose rollout succeeded (bottle in the box, see rollout_physics.py).
Episodes that failed in the pipeline (e.g. hand out of view, no grasp found) have
no demo file and are listed as excluded.

Layout (robomimic convention):

  data/                      attrs: total, env_args (JSON)
    demo_0/                  attrs: num_samples, model_file, clip, control_freq, ...
      actions        (N, 7)  delta end-effector: position (m), rotation axis-angle (rad),
                             gripper (-1 open, +1 close); target of the step minus current pose
      actions_abs    (N, 7)  absolute end-effector target: position, axis-angle, gripper
      actions_joint  (N, 8)  joint position targets (7) + gripper; what the sim executed
      rewards, dones (N,)    reward 1 while the bottle is in the box; done on the last step
      states         (N, 1+nq+nv)  MuJoCo state [time, qpos, qvel] (to reset the sim)
      obs/, next_obs/        agentview_image, robot0_eye_in_hand_image (uint8 HxWx3),
                             robot0_joint_pos/_vel, robot0_eef_pos, robot0_eef_quat (xyzw),
                             robot0_gripper_qpos/_qvel, object (13)
  mask/train, mask/valid     demo names for a train / validation split

The environment is our own MuJoCo scene (not a robosuite env), so train with
rollouts disabled (experiment.rollout.enabled = false) unless an evaluation env
is written for it.

Usage:
    uv run pipeline/make_robomimic.py
    uv run pipeline/make_robomimic.py --valid-ratio 0.1 --out videos/task/dataset_sim/robomimic/pick_place_bottle.hdf5
"""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).parent.parent
SIM = ROOT / "videos" / "task" / "dataset_sim"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", default=str(SIM / "episodes"), help="folder with one subfolder per episode")
    p.add_argument("--out", default=str(SIM / "robomimic" / "pick_place_bottle.hdf5"))
    p.add_argument("--valid-ratio", type=float, default=0.1, help="share of demos in the validation mask")
    p.add_argument("--require-upright", action="store_true", help="also drop episodes where the bottle fell over")
    p.add_argument("--seed", type=int, default=0, help="seed of the train / valid split")
    return p.parse_args()


def main():
    args = parse_args()
    episodes = sorted(p for p in Path(args.episodes).iterdir() if p.is_dir())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept, report = [], []
    for ep in episodes:
        demo, phys = ep / "demo.hdf5", ep / "physics.json"
        if not demo.exists() or not phys.exists():
            report.append({"episode": ep.name, "kept": False, "reason": "no physics rollout (pipeline failed)"})
            continue
        res = json.loads(phys.read_text())
        if not res.get("success"):
            why = "bottle not in the box" if not res.get("inside_container") else (
                "bottle still in the gripper" if res.get("touching_gripper") else "bottle not touching the box")
            report.append({"episode": ep.name, "kept": False, "reason": why})
            continue
        if args.require_upright and res["bottle_tilt_deg"] >= 20:
            report.append({"episode": ep.name, "kept": False, "reason": "bottle fell over in the box"})
            continue
        kept.append(ep)
        report.append({"episode": ep.name, "kept": True, "reason": ""})

    if not kept:
        raise SystemExit("No successful episode to export.")

    total = 0
    with h5py.File(out, "w") as f:
        data = f.create_group("data")
        names = []
        for i, ep in enumerate(kept):
            name = f"demo_{i}"
            with h5py.File(ep / "demo.hdf5", "r") as src:
                for key in src:
                    src.copy(src[key], data, name=f"{name}/{key}")
                for k, v in src.attrs.items():
                    data[name].attrs[k] = v
                data[name].attrs["episode"] = ep.name
                total += int(src.attrs["num_samples"])
                control_freq = float(src.attrs["control_freq"])
            names.append(name)
        data.attrs["total"] = total
        data.attrs["env_args"] = json.dumps({
            "env_name": "PandaPickPlaceBottle",
            "type": 2,  # not a robosuite env: our MuJoCo scene (model_file in each demo)
            "env_kwargs": {
                "simulator": "mujoco",
                "control_freq": control_freq,
                "robot": "Franka Panda (MuJoCo Menagerie)",
                "camera_names": ["agentview", "panda/wrist"],
                "action_space": "delta end-effector (actions), absolute end-effector (actions_abs), "
                                "joint positions (actions_joint)",
                "success": "bottle touches the container, centre inside its footprint, released",
            },
        })
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(names))
        n_valid = int(round(args.valid_ratio * len(names))) if len(names) > 1 else 0
        valid = sorted(names[j] for j in order[:n_valid])
        train = sorted(names[j] for j in order[n_valid:])
        mask = f.create_group("mask")
        mask.create_dataset("train", data=np.array(train, dtype="S"))
        mask.create_dataset("valid", data=np.array(valid, dtype="S"))

    report_path = out.with_name(out.stem + "_episodes.csv")
    with open(report_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["episode", "kept", "reason"])
        w.writeheader()
        w.writerows(report)

    print(f"{len(kept)} of {len(episodes)} episodes kept ({total} samples), "
          f"{len(train)} train / {len(valid)} valid")
    for r in report:
        if not r["kept"]:
            print(f"  excluded {r['episode']}: {r['reason']}")
    print(f"Saved {out}\nSaved {report_path}")


if __name__ == "__main__":
    main()
