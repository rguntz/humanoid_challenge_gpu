"""Convert the merged robomimic dataset into the policy's training dataset.

Input: videos/task/dataset_sim/robomimic/pick_place_bottle.hdf5 (pipeline/make_robomimic.py),
whose observations and actions are in each episode's own scene (tag) frame.

Output, in --out-dir (default videos/task/dataset_sim/policy/):
  pick_place_bottle_base.hdf5   same robomimic layout, but
      obs/      robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, object in the
                robot-base frame (see task.py), agentview_image unchanged
      actions   the absolute end-effector target (recorded actions_abs) in the base frame
      layout/   one row per demo: start joint positions, bottle and container x/y,
                table and tag placement, all in the base frame (env.py samples
                its initial states from these)
  env_scene.xml, tag_texture.png   one episode's scene, for the evaluation env

Each demo is cut --steps-after-success steps (default 8, about 0.5 s) after its
first success (bottle released in the container, reward 1). The recordings go on
until the demonstrator's hand is back at rest (~3 s, ~30 cm of arm motion, 28 % of
the steps): never executed by the policy, since the env ends an episode on success,
and every action chunk starting near the release would otherwise include that
pull-back. The 8 steps keep the release and a short move away (one execution
horizon) in the data.

Check: the first step of every demo is re-observed from its saved MuJoCo state
with the env's functions (task.py) and must match the converted observation. Only
the first step is exact: the recorder reads positions right after mj_step, which
leaves them one physics step (2 ms) behind the saved state; the env reads them the
same way, so its observations match the recorded ones. The later steps are compared
too, as a bound (a fraction of a millimetre while the arm moves).

Usage:
    uv run policy/make_dataset.py
"""
import argparse
import json
import os
import re
import shutil
from pathlib import Path

import h5py
import mujoco
import numpy as np

import task

SIM = task.ROOT / "videos" / "task" / "dataset_sim"
LAYOUT_KEYS = ("robot_qpos", "bottle_xy", "box_xy", "table_center", "table_half", "tag_pos")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(SIM / "robomimic" / "pick_place_bottle.hdf5"))
    p.add_argument("--episodes", default=str(SIM / "episodes"), help="episode folders (for each demo's scene.xml)")
    p.add_argument("--out-dir", default=str(SIM / "policy"))
    p.add_argument("--camera", default="agentview_image", help="image key to keep")
    p.add_argument("--steps-after-success", type=int, default=8,
                   help="keep this many steps after a demo's first success, cut the rest (-1: keep everything)")
    return p.parse_args()


def set_state(model, data, state):
    """states rows are [time, qpos, qvel] (rollout_physics.Recorder)."""
    data.time = state[0]
    data.qpos[:] = state[1:1 + model.nq]
    data.qvel[:] = state[1 + model.nq:1 + model.nq + model.nv]
    mujoco.mj_forward(model, data)


def write_env_scene(model_file, episode_dir, out_dir):
    """The scene of one episode, with its mesh folder made relative to out_dir."""
    meshdir = re.search(r'meshdir="([^"]*)"', model_file).group(1)
    assets = (episode_dir / meshdir).resolve()
    rel = os.path.relpath(assets, out_dir.resolve())
    (out_dir / "env_scene.xml").write_text(model_file.replace(f'meshdir="{meshdir}"', f'meshdir="{rel}/"'))
    shutil.copy(episode_dir / "tag_texture.png", out_dir / "tag_texture.png")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "pick_place_bottle_base.hdf5"
    episodes = Path(args.episodes)

    layouts = {k: [] for k in LAYOUT_KEYS}
    yaws, kept_steps, all_steps = [], 0, 0
    first_err, later_err = ({k: 0.0 for k in task.LOWDIM_KEYS} for _ in range(2))
    with h5py.File(args.src, "r") as src, h5py.File(out, "w") as dst:
        names = sorted(src["data"], key=lambda n: int(n.split("_")[1]))
        data_out = dst.create_group("data")
        for k, v in src["data"].attrs.items():
            data_out.attrs[k] = v
        for name in names:
            d = src["data"][name]
            episode_dir = episodes / d.attrs["episode"]
            model = mujoco.MjModel.from_xml_path(str(episode_dir / "scene.xml"))
            mdata = mujoco.MjData(model)
            scene = task.Scene(model)
            # Cut the demo a few steps after its first success (see the module docstring).
            T = len(d["rewards"])
            rewards = d["rewards"][:]
            if args.steps_after_success >= 0 and (rewards > 0).any():
                T_keep = min(T, int(np.argmax(rewards > 0)) + args.steps_after_success + 1)
            else:
                T_keep = T
            all_steps, kept_steps = all_steps + T, kept_steps + T_keep
            states = d["states"][:T_keep]
            set_state(model, mdata, states[0])
            pb, Rb = task.base_pose(model, mdata)
            yaws.append(task.base_yaw_deg(Rb))
            for k, v in task.layout(model, mdata, scene).items():
                layouts[k].append(v)
            if name == names[0]:
                write_env_scene(d.attrs["model_file"], episode_dir, out_dir)

            g = data_out.create_group(name)
            for k, v in d.attrs.items():
                g.attrs[k] = v
            g.attrs["num_samples"] = T_keep
            g.attrs["num_samples_recorded"] = T
            obs = task.obs_to_base({k: d["obs"][k][:T_keep] for k in task.LOWDIM_KEYS}, pb, Rb)
            for k, v in obs.items():
                g.create_dataset(f"obs/{k}", data=v)
            img = d["obs"][args.camera]
            g.create_dataset(f"obs/{args.camera}", data=img[:T_keep], compression="gzip", compression_opts=4,
                             chunks=(1,) + img.shape[1:])
            g.create_dataset("actions", data=task.actions_to_base(d["actions_abs"][:T_keep], pb, Rb))
            dones = np.zeros(T_keep, dtype=d["dones"].dtype)
            dones[-1] = 1
            g.create_dataset("rewards", data=rewards[:T_keep])
            g.create_dataset("dones", data=dones)
            g.create_dataset("states", data=states)

            # Check (see the module docstring): re-observe from the saved states.
            for t, s in enumerate(states):
                set_state(model, mdata, s)
                live = task.obs_to_base(task.world_lowdim_obs(mdata, scene), pb, Rb)
                err = first_err if t == 0 else later_err
                for k in task.LOWDIM_KEYS:
                    err[k] = max(err[k], float(np.abs(live[k] - obs[k][t]).max()))

        data_out.attrs["total"] = kept_steps
        data_out.attrs["steps_after_success"] = args.steps_after_success
        lay = dst.create_group("layout")
        for k, v in layouts.items():
            lay.create_dataset(k, data=np.array(v, dtype=np.float64))
        lay.attrs["demos"] = json.dumps(names)
        if "mask" in src:
            src.copy(src["mask"], dst, name="mask")

    print(f"{len(names)} demos -> {out}")
    print(f"steps kept: {kept_steps} of {all_steps} ({100 * kept_steps / all_steps:.0f} %), "
          f"cut {args.steps_after_success} steps after the first success")
    print(f"base yaw (deg): {sorted(set(np.round(yaws, 3)))}")
    if np.ptp(yaws) > 1e-3:
        raise SystemExit("The robot base yaw differs between episodes: the table/tag layout transform "
                         "in task.layout assumes it does not.")
    for k in ("bottle_xy", "box_xy"):
        v = np.array(layouts[k])
        print(f"{k:10s} base frame, m: x {v[:, 0].min():.3f} .. {v[:, 0].max():.3f}, "
              f"y {v[:, 1].min():.3f} .. {v[:, 1].max():.3f}")
    for label, err in (("first step (must be ~0)", first_err), ("later steps (one 2 ms physics step)", later_err)):
        print(f"re-observed vs converted, {label}: " + ", ".join(f"{k} {e:.1e}" for k, e in err.items()))
    if max(first_err.values()) > 1e-4:
        raise SystemExit("The first-step observations do not match: the conversion and the env disagree.")
    print(f"Saved {out_dir / 'env_scene.xml'}")


if __name__ == "__main__":
    main()
