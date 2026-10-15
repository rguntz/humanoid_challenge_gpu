# Humanoid challenge

## Setup

Tested setup: Ubuntu-like Linux, x86_64, NVIDIA driver recent enough for CUDA 12.8
(`nvidia-smi` shows "CUDA Version: 12.8" or higher, i.e. driver 570+). Hand tracking
(below) uses the GPU automatically if there is one; rendering is headless (MuJoCo uses
EGL when there is no display).

**1. Get the project**: clone this repository.

**2. System packages** (Ubuntu / Debian; the libraries by OpenCV and by MuJoCo's
headless rendering, and FFmpeg's shared libraries (version 4 to 7) by torchcodec,
which decodes the videos of the LeRobot datasets used by `resfit/`):

```sh
sudo apt update && sudo apt install -y curl libgl1 libglib2.0-0 libegl1 ffmpeg
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

**5. Download the Panda model**: the evaluation env needs the Franka Panda mesh
assets in `sim/third_party/` (the full pipeline below would also fetch them on its
own first run, but this gets it out of the way upfront):

```sh
uv run python -c "import sys; sys.path.insert(0, 'sim'); from build_scene import ensure_panda; ensure_panda()"
```

This sparse-clones only the `franka_emika_panda/` folder from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie); the
download is ~10 MB.

## Recording your own demonstrations

Everything below starts from one raw egocentric video per recording session, placed
in `videos/task/dataset/` (the folder is not in git: there is nothing there until you
record and drop a video in). To produce one:

1. Print the fiducial tag (`tag/tag_5x5_id0_150mm.png`, already sized to 150 mm) and
   place it flat on the table in view of the camera: it is how the pipeline later
   recovers the camera/table pose for each episode.
2. `calibration/camera_intrinsics.json` is already in the repository, calibrated for
   the phone this task was recorded with. If you record with a different phone,
   recalibrate it first: print `tag/checkerboard_10x7_20mm_A4.pdf`, then film it with
   that phone for at least some tens of seconds while continuously moving it around,
   varying its position in frame, its distance from the camera, and its tilt/angle.
   `calibrate_camera.py` automatically picks a diverse subset of frames from the video
   (different positions, distances and tilts) to solve for the lens's focal length and
   distortion; a video where the board barely moves does not give it enough different
   viewpoints to calibrate accurately. Then:
   `uv run calibration/calibrate_camera.py <that video>`.
3. Record one continuous egocentric video of as many pick-and-place demonstrations as
   you want, back to back, resetting the bottle and container between each. Start and
   end every single demonstration with about 1 s completely still, tag visible and
   hand flat on the table: the tag-pose and hand-tracking steps of the pipeline rely
   on that still moment to calibrate against.
4. Save the video into `videos/task/dataset/`.

Then use the episode cutter below to split it into individual episode clips.

## Episode cutter (video_editing)

Takes one long egocentric recording (multiple demonstration episodes back to back,
with a reset in between each) from `videos/task/` and cuts it into individual
per-episode clips in `videos/task/dataset/`.

```sh
uv run video_editing/server.py
```

A local dashboard opens at http://localhost:8765 (use Chrome or Safari: the phone
videos are HEVC, which Firefox cannot play). Mark each episode's start/end in the
timeline, then export; the tool is self-explanatory from there.

## From episode videos to a robomimic dataset (pipeline)

```sh
uv run pipeline/run_dataset.py
```

The first run downloads the WiLoR weights (~2.6 GB, into
`hand_tracking/pretrained_models/`) and the Panda model (into `sim/third_party/`):
the machine needs internet access then (or copy these two folders over). For a long
run over SSH, use `tmux` or `nohup ... &` so it survives a disconnection; it can be
restarted at any time, finished episodes are skipped. WiLoR-mini downloads the MANO
hand model from a third-party mirror; MANO is licensed for non-commercial research
(registration at mano.is.tue.mpg.de). Hand tracking is the slow step (~0.6 s per frame
on a Mac, much faster on CUDA).

For every clip in `videos/task/dataset/`, this turns the raw egocentric recording of a
human demonstration into a simulated robot trajectory, by running:

1. **Tag pose**: in the real video's first still frame, locate the fiducial marker
   printed on the table to recover the phone camera's pose relative to it. This real-
   world camera/table pose is what the simulated scene (step 5) is then built to match.
2. **Hand tracking (WiLoR)**: track the hand's 3D pose and finger articulation frame
   by frame from the video alone.
3. **Virtual gripper**: turn the tracked hand into an equivalent parallel-jaw gripper
   pose and open/close state (humans don't have a gripper, so this is a mapping, not
   a direct reading).
4. **Depth correction**: the monocular hand pose is depth-ambiguous; this corrects it
   using the known table plane and object contact.
5. **MuJoCo scene**: build the simulated scene for this episode (table, Panda arm,
   tag, phone camera, bottle, container) placed consistently with the tracked poses.
6. **IK**: solve inverse kinematics so the simulated Panda arm's end-effector follows
   the virtual gripper's tracked trajectory.
7. **Physics rollout**: actually simulate the resulting joint trajectory in MuJoCo,
   producing the robot/object states and observations that become the dataset.

Outputs in `videos/task/dataset_sim/`:

| Path | Content |
|---|---|
| `videos/<clip>.mp4` | real video with the simulated robot drawn over it, and a side view |
| `episodes/<clip>/` | all intermediate files, `demo.hdf5` (the recorded rollout), `log.txt` |
| `summary.csv` | one row per episode: status, failing step, success, key numbers |
| `robomimic/pick_place_bottle.hdf5` | the training dataset: successful episodes only |
| `robomimic/pick_place_bottle_episodes.csv` | which episodes were kept or excluded, and why |

**Success** (only these episodes go into the dataset): at the end of the rollout the
bottle touches the container (floor or inside of the walls).

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

## Converting to the robot-base frame and LeRobot datasets

Shared by both training paths below ([policy/](#training-a-policy-policy) and
[resfit/](#training-with-resfit-resfit)): turns the robomimic dataset above into the
robot-base-frame dataset `policy/`'s own environment reads, and into the LeRobot
datasets `resfit/` trains on.

**1. Convert the dataset** to the robot-base frame (once, after `run_dataset.py`):

```sh
uv run resfit/pick_place/make_dataset.py
```

This writes the robot-base dataset (trajectories smoothed with sigma 0.8, per-demo
videos), then the LeRobot datasets `resfit/` trains on, one per action mode, in
`videos/task/dataset_sim/lerobot/`. Each is checked by replaying it in the simulation;
the replays are saved in its `check_videos/` folder.

Every LeRobot dataset is named `<version>_<action mode>`. Each episode has
`observation.state` (end-effector position, quaternion xyzw, 2 finger joints - not
joint angles), `observation.images.agentview` and `observation.images.wrist`
(128 x 128 video), `action` and `next.done`, and ends at the demo's first success.

| Action mode | Size | Action |
|---|---|---|
| `absolute` | 7 | end-effector target: position, rotation axis-angle, gripper |
| `delta` | 7 | the same target relative to the current end-effector pose |
| `absolute_6d` | 10 | position, rotation as the first two rows of its matrix (continuous, unlike axis-angle), gripper |
| `joint` | 8 | the demo's recorded joint targets (7 arm joints) and gripper, executed without IK; `observation.state` is then the 7 arm joints and 2 fingers |

Each episode's scene is in the tag frame, with the robot base placed behind that
episode's workspace, so the same arm pose has different coordinates in different
episodes. The conversion expresses everything relative to the robot base instead.

It also writes the same converted demos as a robomimic-style HDF5,
`videos/task/dataset_sim/policy/pick_place_bottle_base.hdf5`, for `policy/`'s own use
(see [policy/README.md](policy/README.md)) — the same per-step content as the LeRobot
datasets above, plus two things the LeRobot datasets don't have: the ground-truth
bottle and container positions (`obs/object`), and a `layout/` table of every demo's
start state, from which `policy/`'s evaluation env draws its random resets.
`env_scene.xml`/`tag_texture.png` next to it are the scene that evaluation env runs.

Each demo is also cut 8 steps (about 0.5 s) after its first success (bottle
released in the container): the recordings go on until the demonstrator's hand is
back at rest, about 3 s and 30 cm of pull-back per demo (28 % of all steps), which
the policy never executes (an evaluation episode ends on success) and which would
otherwise leak into every action chunk around the release. `--steps-after-success N`
changes the 8 (`-1` keeps everything).

It prints a check: the first step of every demo, re-observed from its saved MuJoCo
state with the env's own code, must match the converted observation (~1e-7).

### Dataset versions

| Version | What differs | Built by |
|---|---|---|
| `pick_place_bottle` | the demos as converted | `make_dataset.py` (above) |
| `pick_place_bottle_full` | the demos' full episodes: the frames after the first success are kept, as `policy/` trains on them (behaviour cloning only) | `convert_pick_place_to_lerobot.py --full-episodes --name pick_place_bottle_full` |

Joint actions replay every demo of `pick_place_bottle` (59/59, 55 at the recorded step and 4
one step later): they are the recording's own control, and need no IK.

Delta actions applied from the live pose drift in `pick_place_bottle` (26/59 replays
succeed: the simulator reaches targets through IK, the demos were recorded with joint
targets).

## Training a policy (policy)

Chunked imitation-learning policies (a simple chunked BC, flow matching and Diffusion
Policy) in JAX / Flax, ported from the xqcfd project. This was the initial training
path for this task; **results were not as good as [ResFiT](#training-with-resfit-resfit)
below**, which is what this repo is actually submitted with — `policy/` is kept only
for reference. See [policy/README.md](policy/README.md) for setup and usage (it reads
the `pick_place_bottle_base.hdf5` dataset from the conversion step above).

## Training with ResFiT (resfit)

`resfit/` is a port of [ResFiT](https://github.com/amazon-science/residual-offpolicy-rl)
(residual off-policy RL on top of a behaviour-cloning policy) to this task, in PyTorch
with LeRobot. Its behaviour-cloning stage is ported: it trains a base policy on the
LeRobot datasets and evaluates it in the pick-and-place MuJoCo simulation
(`resfit/pick_place/env.py`), resetting from the recorded demos' own start states. The
residual RL stage is ported too (see Residual RL below).

See [Converting to the robot-base frame and LeRobot datasets](#converting-to-the-robot-base-frame-and-lerobot-datasets)
above for the dataset formats and versions `resfit/` trains on.

### Training

```sh
bash resfit/lerobot/shell/train_bc.sh                       # defaults: ACT, absolute actions
POLICY=smolvla ACTION_MODE=absolute_6d GPU=1 bash resfit/lerobot/shell/train_bc.sh
bash resfit/lerobot/shell/train_bc_pi0_absolute_6d.sh       # a ready-made setup
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
| `smolvla` | LeRobot's SmolVLA vision-language-action model (450M parameters, flow-matching action expert), one observation frame, the task sentence as instruction; `PRETRAINED=lerobot/smolvla_base` starts from the pretrained model (vision-language model frozen, action expert trained) |
| `pi0` | LeRobot's PI0 vision-language-action model (PaliGemma-3B backbone + flow-matching action expert), one observation frame, the task sentence as instruction; `PRETRAINED=lerobot/pi0` starts from the pretrained checkpoint (its own bf16 precision kept as released, not forced to fp32 like SmolVLA) |

| Setting | Default | Meaning |
|---|---|---|
| `DATASET`, `ACTION_MODE` | `pick_place_bottle`, `absolute` | which LeRobot dataset ([Dataset versions](#dataset-versions)) |
| `CAMERAS` | all | `"agentview wrist"`, `agentview` or `wrist` |
| `N_OBS_STEPS` (To) | the policy's own (ACT, MLP: 1; Diffusion Policy family: 2) | observation frames |
| `CHUNK_SIZE` (Ta) | 16 | actions predicted per call |
| `N_ACTION_STEPS` | 8 | execution horizon: actions executed per call |
| `NORMALIZER` | `mean_std` | `per_dim_range`: state and position actions mapped to [-1, 1] per dimension, quaternion / rotation / gripper unchanged |

Every other setting (batch size, optimizer, precision, logging, evaluation cadence, ...)
is documented in `train_bc.sh`'s own header comments.

Each run writes to `outputs/resfit_bc/<run name>/`: `latest/` (to resume), `best/` (the
best evaluation so far, with EMA its averaged weights: the policy the residual RL will
start from) and `evals/` (one video per evaluation, both cameras side by side).

Each run takes about 28 GB of RAM (data workers and evaluation simulators). On a shared
machine, systemd-oomd kills the whole user session (tmux included) when memory runs out:
check `free -g` before starting several runs.

Checks: `resfit/pick_place/check_vec_env.py` (the vectorized env against the datasets)
and `resfit/pick_place/check_rollout_obs.py` (the simulator's observations against the
dataset's along every replayed demo).

### Evaluating a checkpoint

```sh
bash resfit/lerobot/shell/eval_bc.sh             # defaults: the SmolVLA unfrozen run's best/, demo layouts, on GPU 0
RUN=outputs/resfit_bc/<run name> N_ACTION_STEPS="16 8 4" bash resfit/lerobot/shell/eval_bc.sh
```

Evaluates a saved checkpoint without training, at several execution horizons, writing
videos and per-episode/per-horizon results to `<RUN>/evals_offline/`. Works for every
policy `train_bc.py` saves.

### Residual RL

```sh
bash resfit/rl_finetuning/shell/train_residual_td3.sh
```

ResFiT's residual TD3 (`resfit/rl_finetuning/scripts/train_residual_td3.py`): a small
actor-critic adds a correction of at most `ACTION_SCALE` on top of a frozen `train_bc.py`
policy's action, trained on a mix of its own transitions and the demos'. Writes to
`outputs/resfit_residual/<run name>/`.

