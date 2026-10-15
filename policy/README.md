# Policy (`policy/`)

Chunked imitation-learning policies (a simple chunked BC, flow matching and Diffusion
Policy), trained on the robomimic dataset and evaluated in the same MuJoCo simulation.
The learning code in `policy/xqc/` comes from the xqcfd project (JAX / Flax), with
its agent, dataset windowing (observation history `To`, action chunk `Ta`, executed
`execution_horizon` steps at a time), normalisation, EMA and evaluation loop.

**This was the initial training path for this task. Results were not as good as
[ResFiT](../README.md#training-with-resfit-resfit)**, which is what the repo is
actually submitted with — this folder is kept only for reference, not actively
developed further.

Before training here, follow the main [README](../README.md)'s **Episode cutter**,
**From episode videos to a robomimic dataset**, and **Converting to the robot-base
frame and LeRobot datasets** sections first — this folder's own training reads
`videos/task/dataset_sim/policy/pick_place_bottle_base.hdf5`, produced by that last
step (`uv run resfit/pick_place/make_dataset.py`).

## Train

From the repository root:

```sh
uv run policy/train.py agent=bc_vision_abs     # chunked BC: quick check of the whole pipeline
CUDA_VISIBLE_DEVICES=0  uv run policy/train.py agent=fm_vision_abs     # flow matching: same network as DP, 10 Euler steps to sample
CUDA_VISIBLE_DEVICES=0 uv run policy/train.py                         # Diffusion Policy (agent=dp_vision_abs, the default)
```

All use the agentview camera plus the robot state (no ground-truth object pose),
absolute end-effector actions, and train the actor only by behaviour cloning
(`critic_name: null`, `max_steps: 0`; the online-RL loop of the script is kept for
later). Settings are Hydra configs in `policy/conf/` and can be overridden on the
command line, for example:

```sh
CUDA_VISIBLE_DEVICES=2 uv run policy/train.py agent=bc_vision_abs \
    agent.actor_pre_training_steps=100000 eval_interval=5000 \
    wandb.mode=offline
```

| Setting | Default | Meaning |
|---|---|---|
| `agent` | `dp_vision_abs` | `dp_vision_abs`, `fm_vision_abs` or `bc_vision_abs` |
| `agent.actor_pre_training_steps` | 100000 | gradient steps |
| `eval_interval` | 10000 | steps between evaluations |
| `eval_num_seeds` × `eval_episodes` | 10 × 5 | parallel env processes × episodes each |
| `save_dir` | `checkpoints/<agent>/<date>_<time>/` | a new folder per launch: checkpoints, one per evaluation (last 3 kept) + `best`, and the eval videos |
| `wandb.mode` | online | `offline` or `disabled` to not upload |
| `eval_video` | true | record eval episodes; saved as mp4 in `<save_dir>/eval_videos/step_<N>/env<i>.mp4` (`eval_video_local`), not uploaded to wandb unless `eval_video_wandb=true` |
| `val_ratio` | 0.02 | share of demos held out for `val_loss` / `val_action_mse` |

**Evaluation env** (`policy/xqc/envs/pick_place.py`): the same simulation as the
demos (position actuators tracking interpolated joint targets at 15 Hz, gripper
fully closed or open, same success test). Each action goes through the same IK as
the demos (`sim/replay_hand.py`). An episode ends on success (bottle released in
the container) or after 1.5 x the longest demo. Each reset draws a new start state
from what the demos cover, in the base frame:

- robot: the start joint positions of a random demo;
- bottle and container: random points in the regions (convex hulls) covered by the
  demos' bottle and container positions, at least as far apart as the closest pair
  in the demos;
- table and tag: where a random demo had them (they show in the camera image).

Replaying the recorded actions of every demo from its own start state in this env
puts the bottle in the box for all 59 demos.

Two env options (`xqc/envs/pick_place.py`'s `PickPlaceBottleEnv`, both default
`false`, set via `env.simplified_init=true` / `env.terminate_on_drop=true` on the
command line):

- `simplified_init`: narrows the robot start pose to a small neighborhood around one
  canonical pose and fixes the container at one canonical position — only the pill
  box keeps moving across its full range. Meant to isolate whether the combinatorial
  spread of start states (not the task itself) is behind a policy not improving.
- `terminate_on_drop`: ends the episode the moment the pill box's cap touches the
  table (tipped over or fallen, checked even if it never leaves the table), instead
  of running out the full horizon with a dropped box sitting uselessly in the replay
  buffer for the rest of the episode.

The training dependencies (JAX 0.4.30 with CUDA 12, Flax, Optax, Orbax,
TensorFlow Probability, Hydra, wandb) are in the repo's `pyproject.toml`, so `uv sync`
from the repository root installs them too.

## Residual RL fine-tuning

After training a base policy (FM recommended), you can fine-tune a step-level
residual policy on top of it using off-policy RL:

```bash
CUDA_VISIBLE_DEVICES=1 uv run policy/train_residual.py
```

The default config (`policy/conf/residual_config.yaml`) already points
`base_checkpoint` at the latest FM checkpoint. To use a different one:

```bash
CUDA_VISIBLE_DEVICES=0 uv run policy/train_residual.py \
    base_checkpoint=checkpoints/FM/2026-10-06_20-32-14/best
```

**What it does:**

1. Loads the frozen FM base policy and its ResNet vision encoder.
2. Pre-populates an offline replay buffer by running the base policy on every
   demo and storing `(obs_window, norm_base, GT_action, reward, next)` transitions.
3. Runs an online RL loop: at each step the residual MLP predicts a 7D delta
   on top of the base policy's action; the combined action is sent to the env.
4. Trains a VMapCritic (C51, 2-ensemble) on mixed offline + online batches,
   then trains the residual actor to maximise critic Q.

Key hyperparameters (all overridable on the CLI):

| Parameter | Default | Meaning |
|---|---|---|
| `total_steps` | 300 000 | total env steps |
| `learning_starts` | 5 000 | random-noise warmup before updates |
| `critic_warmup_steps` | 10 000 | critic-only steps before actor updates |
| `offline_fraction` | 0.5 | fraction of each batch from offline buffer |
| `residual.action_scale` | 0.1 | max residual magnitude per action dim |
| `residual.hidden_dims` | [256, 256] | residual MLP hidden layers |
| `critic.hidden_dims` | [256, 256, 256] | critic MLP hidden layers |
