"""Check the vectorized env (vec_env.py) against the LeRobot datasets.

For every action mode's dataset, every episode's actions are read from the
dataset and sent through create_vectorized_env (two simulators in parallel, in their
own processes, both started from the demo's layout), the way a policy's actions are.
Required:
  1. Same outcome as the direct replay from the live pose (the replay the converter
     reports, convert_pick_place_to_lerobot.py): success at the same step, or no
     success. Absolute actions reach success in every episode; delta actions only
     in some, without a policy to correct the drift (see the converter).
  2. Both parallel simulators give the same result.
  3. The step that ends an episode reports it as the vector env must:
     terminated, info["final_info"] with "success" and "episode_steps", and
     info["final_obs"].
  4. The first observation has the dataset's features, shapes and dtypes, and equals
     the episode's first frame: state exactly, images up to video compression.
  5. render() gives both cameras side by side for every env.

Usage:
    uv run resfit/pick_place/check_vec_env.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from resfit.lerobot.dataset.convert_pick_place_to_lerobot import HOLD_STEPS, replay  # noqa: E402
from resfit.pick_place.action_mode import ACTION_MODES  # noqa: E402
from resfit.pick_place.env import PickPlaceBottleEnv  # noqa: E402
from resfit.pick_place.vec_env import create_vectorized_env, dataset_env_config  # noqa: E402

LEROBOT = REPO / "videos" / "task" / "dataset_sim" / "lerobot"
NUM_ENVS = 2
IMAGE_TOLERANCE = 0.03  # env image vs decoded dataset frame, mean abs error (render + video compression)


def run_episode(vec_env, layout_row, actions):
    """Send one episode's actions (then the last one, HOLD_STEPS more times) to every
    env. Returns per env the step of the first success (None if none), the first
    observation, and the problems found in the end-of-episode info."""
    obs, _ = vec_env.reset(options={"demo": layout_row})
    first_obs = {k: v.cpu().numpy() for k, v in obs.items()}
    n = len(actions)
    success_step, problems = [None] * NUM_ENVS, []
    done = np.zeros(NUM_ENVS, bool)
    for t in range(n + HOLD_STEPS):
        batch = np.repeat(actions[min(t, n - 1)][None], NUM_ENVS, axis=0)
        _, reward, terminated, truncated, info = vec_env.step(batch)
        for i in np.flatnonzero((terminated | truncated).cpu().numpy() & ~done):
            done[i] = True
            final = info.get("final_info", {})
            if terminated[i]:
                success_step[i] = t
                if not (final.get("_success", [False] * NUM_ENVS)[i] and final["success"][i]):
                    problems.append(f"env {i}: terminated without final_info success")
                if reward[i] != 1.0:
                    problems.append(f"env {i}: terminated with reward {float(reward[i])}")
            if final.get("episode_steps", [None] * NUM_ENVS)[i] != t + 1:
                problems.append(f"env {i}: final_info episode_steps is not {t + 1}")
            if info.get("final_obs") is None or info["final_obs"][i] is None:
                problems.append(f"env {i}: no final_obs")
        if done.all():
            break
    return success_step, first_obs, problems


def check(mode):
    out = LEROBOT / f"pick_place_bottle_{mode}"
    ds = LeRobotDataset(f"local/pick_place_bottle_{mode}", root=out)
    episodes = json.loads((out / "meta" / "source_demos.json").read_text())
    actions = torch.stack(ds.hf_dataset["action"]).numpy()
    states = torch.stack(ds.hf_dataset["observation.state"]).numpy()
    starts, ends = ds.episode_data_index["from"].tolist(), ds.episode_data_index["to"].tolist()
    obs_keys = [k for k in ds.meta.features if k.startswith("observation.")]

    action_mode, source = dataset_env_config(out)
    assert action_mode == mode, (action_mode, mode)
    plain_env = PickPlaceBottleEnv(source)  # the direct replay, as in the converter
    vec_env = create_vectorized_env(out, num_envs=NUM_ENVS)
    errors, n_success, state_err, image_err = [], 0, 0.0, 0.0
    for ep, i0, i1 in zip(episodes, starts, ends):
        a, s = actions[i0:i1], states[i0:i1]
        expected, _, _, _ = replay(plain_env, ep["layout_row"], a, s, mode, "live", HOLD_STEPS)
        got, first_obs, problems = run_episode(vec_env, ep["layout_row"], a)
        n_success += got[0] is not None
        errors += [f"episode {ep['episode_index']}: {p}" for p in problems]
        if got != [expected] * NUM_ENVS:
            errors.append(f"episode {ep['episode_index']}: success steps {got}, direct replay {expected}")

        frame = ds[i0]
        if sorted(first_obs) != sorted(obs_keys):
            errors.append(f"observation keys {sorted(first_obs)} != dataset's {sorted(obs_keys)}")
        for key in obs_keys:
            ref = frame[key].numpy()
            o = first_obs[key]
            if o.shape != (NUM_ENVS, *ref.shape) or o.dtype != ref.dtype:
                errors.append(f"{key}: {o.dtype} {o.shape}, dataset {ref.dtype} {ref.shape}")
                continue
            err = float(np.abs(o - ref).max() if key == "observation.state" else np.abs(o - ref).mean())
            if key == "observation.state":
                state_err = max(state_err, err)
            else:
                image_err = max(image_err, err)
        print(f"  episode {ep['episode_index']}: success step {got[0]} (direct replay {expected})", flush=True)

    frames = vec_env.render()
    h, w = ds.meta.features["observation.images.agentview"]["shape"][:2]
    if frames.shape != (NUM_ENVS, h, 2 * w, 3):
        errors.append(f"render() gives {frames.shape}, expected {(NUM_ENVS, h, 2 * w, 3)}")
    vec_env.close()
    plain_env.close()

    print(f"{mode}: {len(episodes)} episodes through the vectorized env, {n_success} succeed; "
          f"first observation vs dataset: state max error {state_err:.1e}, images mean error {image_err:.4f}")
    if state_err > 1e-4:
        errors.append(f"first state differs from the dataset's by {state_err:.1e}")
    if image_err > IMAGE_TOLERANCE:
        errors.append(f"first images differ from the dataset's by {image_err:.4f}")
    return errors


def main():
    errors = []
    for mode in ACTION_MODES:
        print(f"=== {mode} actions")
        errors += [f"{mode}: {e}" for e in check(mode)]
    if errors:
        raise SystemExit("Vectorized env check failed:\n" + "\n".join(errors[:30]))
    print("Vectorized env check passed.")


if __name__ == "__main__":
    main()
