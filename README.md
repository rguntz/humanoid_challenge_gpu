# Humanoid challenge

## Episode cutter (video_editing)

A local dashboard to cut a long recording from `videos/task/` into episodes.

```sh
uv run video_editing/server.py
```

This opens http://localhost:8765 in your browser (use Chrome or Safari: the phone
videos are HEVC, which Firefox cannot play). Stop it with `Ctrl+C` in the terminal.
Options: `--port 8765` to change the port, `--no-browser` to not open a tab.

1. Pick the video in the dropdown (the longest one is opened by default).
2. Go to the start of an episode and press **I** (Set start), then go to its end and
   press **O** (Set end). Pressing **I** again later begins the next episode, so
   I → O → I → O … cuts one episode after the other; the gaps where the objects are
   reset are simply left out. To move an existing start or end, click its `s`/`e`
   marker first, then press **I**/**O** at the new frame.
3. Click **Export**. Each episode is cut to `videos/task/dataset/<video>_epNNN.mp4`.

| Key | Action |
|---|---|
| `Space` | play / pause |
| `←` `→` | 1 frame |
| `Shift` + `←` `→` | 1 second |
| `↑` `↓` | 10 seconds |
| `,` / `.` | jump to (and select) the previous / next s/e marker |
| `I` | set start: begins the next episode here (or moves the selected `s` marker, or fills a missing start) |
| `O` | set end of the current episode (or moves the selected `e` marker) |
| `N` | always begin a new episode here |
| `P` | preview the selected episode |
| `Delete` | delete the selected s/e marker (the episode stays, with that end missing) |
| `Shift` + `Delete` | delete the selected episode |

Under the video, two timelines show every start as `s<id>` (green) and every end
as `e<id>` (red), with the cursor: the whole video, and a zoom of ±30 s around the
cursor for frame-level detail. Click a marker to select it (the cursor jumps to it),
click elsewhere to seek. The line above the timelines says where the cursor is,
e.g. `inside episode #4 (s 3:05.10 → e 3:20.00)` or `between e3 and s4`. To fix a
mistake, select the wrong marker, press `Delete`, go to the right frame and press
`I` (start) or `O` (end).

- The episode list is saved as you edit, in
  `videos/task/dataset/<video>_episodes.json`, so you can stop and continue later.
- The original video is never modified.
- Episodes with no end, an end before the start, an overlap with another episode,
  or a length under 2 s are flagged and not exported.
- Export only cuts new or changed episodes (tick **re-export all** to redo them all).
- Clips are frame-accurate (re-encoded with ffmpeg, H.264 CRF 18), upright 1080x1920,
  without audio.
- Each episode should start with about 1 s still (tag visible, hand flat on the
  table) and end with the hand back at rest: the tag pose and the hand calibration
  rely on it.

## From episode videos to a robomimic dataset (pipeline)

```sh
uv run pipeline/run_dataset.py                 # all exported episodes, one after the other
uv run pipeline/run_dataset.py --only 1 2 3    # some episodes (by number)
uv run pipeline/run_dataset.py --no-video      # skip the check videos (faster)
uv run pipeline/run_dataset.py --force         # redo episodes that are already done
```

For every clip in `videos/task/dataset/`, it runs: tag pose → hand tracking (WiLoR)
→ virtual gripper → depth correction → MuJoCo scene (table, Panda, tag, phone
camera, bottle, container) → IK → physics rollout. A failing episode (e.g. hand out
of view, no grasp found) is logged and skipped. The batch can be stopped and
restarted: finished episodes are skipped.

Outputs in `videos/task/dataset_sim/`:

| Path | Content |
|---|---|
| `videos/<clip>.mp4` | real video with the simulated robot drawn over it, and a side view |
| `episodes/<clip>/` | all intermediate files, `demo.hdf5` (the recorded rollout), `log.txt` |
| `summary.csv` | one row per episode: status, failing step, success, key numbers |
| `robomimic/pick_place_bottle.hdf5` | the training dataset: successful episodes only |
| `robomimic/pick_place_bottle_episodes.csv` | which episodes were kept or excluded, and why |

**Success** (only these episodes go into the dataset): at the end of the rollout the
bottle touches the container (floor or inside of the walls), its centre is inside
the container, and it no longer touches the gripper. It does not need to be upright
(`--require-upright` in `make_robomimic.py` drops those too).

**Dataset format** (robomimic HDF5, one `data/demo_i` per episode, recorded at 15 Hz):

- `actions` (N, 7): delta end-effector position (m) + rotation axis-angle (rad) +
  gripper (-1 open, +1 close). Also `actions_abs` (absolute end-effector target) and
  `actions_joint` (7 joint position targets + gripper, what the sim executed).
- `obs` / `next_obs`: `agentview_image` (128x128, fixed camera facing the robot),
  `robot0_eye_in_hand_image` (128x128, wrist camera), `robot0_joint_pos/_vel`,
  `robot0_eef_pos`, `robot0_eef_quat` (xyzw), `robot0_gripper_qpos/_qvel`,
  `object` (bottle position + quaternion, bottle relative to the gripper, container position).
- `rewards` (1 while the bottle is in the box), `dones`, `states` (MuJoCo
  `[time, qpos, qvel]`), and the scene XML in each demo's `model_file` attribute.
- `mask/train`, `mask/valid`: 90 / 10 split.

The environment is our MuJoCo scene, not a robosuite env. To train and evaluate a
policy in it, see [Training a policy](#training-a-policy-policy) below (with robomimic
itself, disable its rollouts: `experiment.rollout.enabled = false`).

### Running it on a Linux machine with an NVIDIA GPU

Hand tracking is the slow step (~0.6 s per frame on a Mac, much faster on CUDA; it
uses the GPU automatically). Rendering is headless (MuJoCo uses EGL when there is no
display). Tested setup: Ubuntu-like Linux, x86_64, NVIDIA driver recent enough for
CUDA 12.8 (`nvidia-smi` shows "CUDA Version: 12.8" or higher, i.e. driver 570+).

**1. Copy the project.** The code (this repository) plus the data that is not in git:

```
calibration/camera_intrinsics.json        # in the repository
videos/task/dataset/*.mp4                 # the episode clips
videos/task/dataset/*_episodes.json       # the episode list written by the cutter
```

**2. System packages** (Ubuntu / Debian; git is needed to install WiLoR-mini and
lerobot from GitHub, the libraries by OpenCV and by MuJoCo's headless rendering, and
FFmpeg's shared libraries (version 4 to 7) by torchcodec, which decodes the videos of
the LeRobot datasets used by `resfit/`):

```sh
sudo apt update && sudo apt install -y git curl libgl1 libglib2.0-0 libegl1 ffmpeg
```

**3. Install uv** (it also installs Python 3.12 by itself):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env        # or open a new terminal
```

**4. Create the environment**, from the project folder:

```sh
cd Humanoid_challenge
uv sync
```

This creates `.venv/` with the exact versions of `uv.lock`, including PyTorch 2.8 with
its CUDA 12.8 libraries (no separate CUDA install needed). No need to activate it:
`uv run ...` always uses it. Use `uv sync --locked` to make it fail instead of
updating `uv.lock` if the lock and `pyproject.toml` ever disagree.

**5. Check the GPU and the headless rendering:**

```sh
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
MUJOCO_GL=egl uv run python -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>'); mujoco.Renderer(m, 64, 64); print('MuJoCo EGL rendering OK')"
```

**6. Run the pipeline:**

```sh
uv run pipeline/run_dataset.py              # or with --no-video to skip the check videos
```

The first run downloads the WiLoR weights (~2.6 GB, into
`hand_tracking/pretrained_models/`) and the Panda model (into `sim/third_party/`):
the machine needs internet access then (or copy these two folders over). For a long
run over SSH, use `tmux` or `nohup ... &` so it survives a disconnection; it can be
restarted at any time, finished episodes are skipped.

Note: WiLoR-mini downloads the MANO hand model from a third-party mirror; MANO is
licensed for non-commercial research (registration at mano.is.tue.mpg.de).

## Training a policy (policy)

Chunked imitation-learning policies (a simple chunked BC, flow matching and Diffusion Policy),
trained on the robomimic dataset above and evaluated in the same MuJoCo simulation.
The learning code in `policy/xqc/` comes from the xqcfd project (JAX / Flax), with
its agent, dataset windowing (observation history `To`, action chunk `Ta`, executed
`execution_horizon` steps at a time), normalisation, EMA and evaluation loop.

**0. Download the Panda model** (if you skipped the pipeline and only have the policy
dataset already, e.g. `videos/task/dataset_sim/policy/` was copied over): the
evaluation env needs the Franka Panda mesh assets in `sim/third_party/`, which the
pipeline would normally fetch on its first run. You can download them standalone:

```sh
uv run python -c "import sys; sys.path.insert(0, 'sim'); from build_scene import ensure_panda; ensure_panda()"
```

This sparse-clones only the `franka_emika_panda/` folder from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) into
`sim/third_party/`. Internet access required; the download is ~10 MB.

**1. Convert the dataset** to the robot-base frame (once, after `run_dataset.py`):

```sh
uv run resfit/pick_place/make_dataset.py
```

This writes the robot-base dataset (trajectories smoothed with sigma 0.8, per-demo
videos), then the LeRobot datasets `resfit/` trains on, one per action mode (absolute,
delta, absolute with 6D rotations), in `videos/task/dataset_sim/lerobot/`. Each is
checked by replaying it in the simulation; the replays are saved in its `check_videos/`
folder. See [Training with ResFiT](#training-with-resfit-resfit) for these datasets and
their regenerated versions.

Each episode's scene is in the tag frame, with the robot base placed behind that
episode's workspace, so the same arm pose has different coordinates in different
episodes. The conversion expresses everything relative to the robot base instead.
It writes to `videos/task/dataset_sim/policy/`:

| Path | Content |
|---|---|
| `pick_place_bottle_base.hdf5` | the training dataset (robomimic layout) |
| `env_scene.xml`, `tag_texture.png` | the scene the evaluation env runs |

In `pick_place_bottle_base.hdf5`:

- `obs/`: `robot0_eef_pos`, `robot0_eef_quat` (xyzw), `robot0_gripper_qpos`, `object`
  in the robot-base frame, and `agentview_image` (unchanged).
- `actions` (N, 7): the absolute end-effector target (`actions_abs`) in the base
  frame: position, rotation axis-angle, gripper (-1 open, +1 close).
- `layout/`: one row per demo with its start state in the base frame (joint
  positions, bottle and container x/y, table and tag placement), from which the
  env draws its initial states.

Each demo is also cut 8 steps (about 0.5 s) after its first success (bottle
released in the container): the recordings go on until the demonstrator's hand is
back at rest, about 3 s and 30 cm of pull-back per demo (28 % of all steps), which
the policy never executes (an evaluation episode ends on success) and which would
otherwise leak into every action chunk around the release. `--steps-after-success N`
changes the 8 (`-1` keeps everything).

It prints a check: the first step of every demo, re-observed from its saved MuJoCo
state with the env's own code, must match the converted observation (~1e-7).

**2. Train** (from the repository root):

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

The training dependencies (JAX 0.4.30 with CUDA 12, Flax, Optax, Orbax,
TensorFlow Probability, Hydra, wandb) are in `pyproject.toml`, so `uv sync` installs
them too. So are those of the residual RL port in `resfit/` (lerobot at a pinned
commit, torchrl 0.9.2, datasets 3.6); see the comments in `pyproject.toml` for why
some versions are pinned or overridden.

### Residual RL fine-tuning

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

## Training with ResFiT (resfit)

`resfit/` is a port of [ResFiT](https://github.com/amazon-science/residual-offpolicy-rl)
(residual off-policy RL on top of a behaviour-cloning policy) to this task, in PyTorch
with LeRobot. Its behaviour-cloning stage is ported: it trains a base policy on the
LeRobot datasets and evaluates it in the same simulation as `policy/` (same env, same
reset protocol). The residual RL stage is not ported yet. Nothing in `resfit/` imports
from `policy/`.

### Datasets

Every LeRobot dataset is named `<version>_<action mode>` and is in
`videos/task/dataset_sim/lerobot/`. Each episode has `observation.state` (end-effector
position, quaternion xyzw, 2 finger joints), `observation.images.agentview` and
`observation.images.wrist` (128 x 128 video), `action` and `next.done`, and ends at the
demo's first success.

| Action mode | Size | Action |
|---|---|---|
| `absolute` | 7 | end-effector target: position, rotation axis-angle, gripper |
| `delta` | 7 | the same target relative to the current end-effector pose |
| `absolute_6d` | 10 | position, rotation as the first two rows of its matrix (continuous, unlike axis-angle), gripper |
| `joint` | 8 | the demo's recorded joint targets (7 arm joints) and gripper, executed without IK; `observation.state` is then the 7 arm joints and 2 fingers |

| Version | What differs | Built by |
|---|---|---|
| `pick_place_bottle` | the demos as converted | `make_dataset.py` (step 1 above) |
| `pick_place_bottle_home` | every episode starts from one home pose (the most central demo start); an approach from it to the demo's start is added | `regenerate_dataset.py --home` |
| `pick_place_bottle_wristcanon` | every demo grasps with the gripper turned the same way (the demos grasp in two groups 180 deg apart about the gripper axis; 13 of 59 are turned) | `regenerate_dataset.py --canonical-wrist` |
| `pick_place_bottle_home_wristcanon` | both | `regenerate_dataset.py --home --canonical-wrist` |

```sh
uv run resfit/pick_place/regenerate_dataset.py --canonical-wrist                 # all action modes
uv run resfit/pick_place/regenerate_dataset.py --home --modes absolute_6d        # one action mode
```

`regenerate_dataset.py` replays every demo in the evaluation env and records new
observations and images (so training images are rendered exactly like the evaluation's),
writes `videos/task/dataset_sim/policy/<version>.hdf5` (the existing datasets are left
as they are), then converts and checks it like `make_dataset.py`. A demo that does not
succeed is left out. The evaluation starts from each version's own start states, so a
policy trained on the home version is also evaluated from the home pose. To redo only
the LeRobot datasets of a version: `uv run resfit/lerobot/dataset/convert_pick_place_to_lerobot.py --src <hdf5> --name <version> [--modes ...]`.

Joint actions replay every demo of `pick_place_bottle` (59/59, 55 at the recorded step and 4
one step later): they are the recording's own control, and need no IK.

Delta actions applied from the live pose drift in `pick_place_bottle` (26/59 replays
succeed: the simulator reaches targets through IK, the demos were recorded with joint
targets); in the regenerated versions they do not (59/59).

### Training

```sh
bash resfit/lerobot/shell/train_bc.sh                       # defaults: ACT, absolute actions
POLICY=legacy_bc ACTION_MODE=absolute_6d GPU=1 bash resfit/lerobot/shell/train_bc.sh
bash resfit/lerobot/shell/train_bc_legacy_bc_absolute_6d.sh # a ready-made setup
```

`train_bc.sh` takes every setting as a variable (see its header); the other
`train_bc_*.sh` files are ready-made setups that set them. W&B logging is required
(`uv run wandb login` once per machine); only metrics and the config go to W&B.

| Policy (`POLICY`) | Model |
|---|---|
| `act` | ResFiT's ACT: ImageNet-pretrained ResNet18 (frozen batch norm), transformer |
| `mlp` | ACT's vision, an MLP predicting the whole chunk instead of the transformer |
| `diffusion` | Diffusion Policy: ResNet18 from scratch (GroupNorm, spatial softmax, random crop), 1D Unet, denoising |
| `unet_bc` | the same Unet and encoder trained by plain BC (MSE on the chunk, one pass) |
| `flow_matching` | the same Unet and encoder trained by flow matching (10 Euler steps to sample) |
| `legacy_bc` | the plain BC of `policy/` (`agent=bc_vision_abs`): MLP 256-256 on Diffusion Policy's encoder, crop 116, chunk predicted from the current step |
| `smolvla` | LeRobot's SmolVLA vision-language-action model (450M parameters, flow-matching action expert), one observation frame, the task sentence as instruction; `PRETRAINED=lerobot/smolvla_base` starts from the pretrained model (vision-language model frozen, action expert trained) |

| Setting | Default | Meaning |
|---|---|---|
| `GPU` | 0 | GPU for training and evaluation rendering |
| `DATASET`, `ACTION_MODE` | `pick_place_bottle`, `absolute` | which LeRobot dataset (see above) |
| `CAMERAS` | all | `"agentview wrist"`, `agentview` or `wrist` |
| `N_OBS_STEPS` (To) | the policy's own (ACT, MLP: 1; Diffusion Policy family: 2) | observation frames |
| `CHUNK_SIZE` (Ta) | 16 | actions predicted per call |
| `N_ACTION_STEPS` | 8 | execution horizon: actions executed per call |
| `NORMALIZER` | `mean_std` | `per_dim_range`: state and position actions mapped to [-1, 1] per dimension, quaternion / rotation / gripper unchanged (the formulation of `policy/`) |
| `BATCH_SIZE`, `STEPS` | 256, 100000 | |
| `OPTIMIZER` | `adamw` (constant learning rate) | `legacy`: betas (0.95, 0.999), 500-step warmup then cosine decay to 0 (as `policy/`); `policy`: the policy's own optimizer and schedule (SmolVLA: betas (0.9, 0.95), 1000-step warmup, cosine decay) |
| `PRETRAINED` | | `smolvla`: checkpoint to start from (folder or Hugging Face id) |
| `GRAD_CLIP_NORM` | 10 | 0: no clipping |
| `EMA` | 0 | 1: evaluate and save an exponential moving average of the weights (decay up to 0.9999) |
| `RANDOM_SHIFT` | 0 | random image shift augmentation, in pixels (not for the Diffusion Policy family, which crops) |
| `PRECISION` | `bf16` | bf16 mixed precision (~1.6-1.9x faster than `fp32` on an RTX 3090) |
| `NUM_WORKERS` | 12 | data loading processes |
| `EVAL_FREQ`, `EVAL_EPISODES` | 10000, 64 | evaluation in 16 simulators from fixed start states (`--eval_seed`) |
| `RUN_NAME` | `<date>_<policy>_<dataset>` | W&B run and output folder |
| `RESUME` | | a run folder to continue |
| `EXTRA_ARGS` | | anything else, e.g. `--policy_kwargs dim_model=256` |

Each run writes to `outputs/resfit_bc/<run name>/`: `latest/` (to resume), `best/` (the
best evaluation so far, with EMA its averaged weights: the policy the residual RL will
start from) and `evals/` (one video per evaluation, both cameras side by side).

Each run takes about 28 GB of RAM (data workers and evaluation simulators). On a shared
machine, systemd-oomd kills the whole user session (tmux included) when memory runs out:
check `free -g` before starting several runs.

Checks: `resfit/pick_place/check_vec_env.py` (the vectorized env against the datasets)
and `resfit/pick_place/check_rollout_obs.py` (the simulator's observations against the
dataset's along every replayed demo).
