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

The environment is our MuJoCo scene, not a robosuite env: train with robomimic's
rollouts disabled (`experiment.rollout.enabled = false`).

### Running it on a Linux machine with an NVIDIA GPU

Hand tracking is the slow step (~0.6 s per frame on a Mac, much faster on CUDA; it
uses the GPU automatically). Rendering is headless (MuJoCo uses EGL when there is no
display). Tested setup: Ubuntu-like Linux, x86_64, NVIDIA driver recent enough for
CUDA 12 (`nvidia-smi` shows "CUDA Version: 12.x" or higher).

**1. Copy the project.** The code (this repository) plus the data that is not in git:

```
calibration/camera_intrinsics.json        # in the repository
videos/task/dataset/*.mp4                 # the episode clips
videos/task/dataset/*_episodes.json       # the episode list written by the cutter
```

**2. System packages** (Ubuntu / Debian; git is needed to install WiLoR-mini from GitHub,
the libraries by OpenCV and by MuJoCo's headless rendering):

```sh
sudo apt update && sudo apt install -y git curl libgl1 libglib2.0-0 libegl1
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

This creates `.venv/` with the exact versions of `uv.lock`, including PyTorch 2.5 with
its CUDA 12 libraries (no separate CUDA install needed). No need to activate it:
`uv run ...` always uses it.

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
