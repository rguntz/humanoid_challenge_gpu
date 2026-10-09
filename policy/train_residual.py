"""Residual RL fine-tuning on top of a pre-trained chunked BC/FM/DP policy.

Architecture:
  - Frozen base policy (loaded from checkpoint) drives a ChunkExecutor
    (To=2, Ta=16, execution_horizon=8).
  - Step-level ResidualMLP predicts a 7D normalised delta every env step.
  - Combined action = clamp(norm_base + norm_delta, -1, 1), unscaled to EEF.
  - Step-level VMapCritic (C51, frozen vision) learns Q(obs, combined_action).

Training phases (matching residual_td3 reference):
  1. Buffer warmup  — collect learning_starts env steps with random noise; no
     gradient updates.
  2. Critic warmup  — critic_warmup_steps pure gradient steps on the filled
     buffers; no env steps; actor stays frozen.
  3. Main RL loop   — env steps + gradient updates; actor fires every
     updates_per_step // actor_updates_per_step critic iterations.

Usage:
    uv run policy/train_residual.py agent=fm_vision_abs \\
        base_checkpoint=checkpoints/FM/DATE/best
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import h5py
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sim"))

import conf.register_envs  # noqa: F401 — registers PickPlaceBottle + resolvers

from xqc.agents.residual.action_scaler import ActionScaler
from xqc.agents.residual.actor import update_actor
from xqc.agents.residual.critic import _encode_frozen, update_critic
from xqc.agents.residual.learner import ObsHistoryWindow, ResidualAgent, jit_encode, jit_residual
from xqc.agents.residual.network import ResidualMLP
from xqc.agents.residual.replay_buffer import (
    OfflineResidualBuffer,
    ResidualReplayBuffer,
    sample_mixed,
)
from xqc.envs.envs import ParallelEnv
from xqc.networks.common import Model, norm_network


# ---------------------------------------------------------------------------
# Agent registry (mirrors train.py's AGENT_REGISTRY)
# ---------------------------------------------------------------------------

def _make_base_agent(cfg: DictConfig, env: ParallelEnv):
    from xqc.agents.actors.bc.bc_learner import BCLearner
    from xqc.agents.actors.dp.dp_learner import DPLearner
    from xqc.agents.actors.fm.fm_learner import FMLearner
    from xqc.replay_buffer.chunked_qc_buffer import load_robomimic_flat

    REGISTRY = {"BC": BCLearner, "DP": DPLearner, "FM": FMLearner}
    AgentClass = REGISTRY[cfg.agent.name]

    offline_data_path = cfg.agent.offline_data_path
    camera_keys = tuple(cfg.agent.camera_keys)
    abs_action = cfg.agent.get("abs_action", False)

    n_demos = cfg.agent.get("n_demos", None)
    if n_demos is None:
        with h5py.File(offline_data_path, "r") as f:
            n_demos = len(f["data"])

    train_dataset, val_dataset, raw_obs_all, raw_actions_all, num_robots, gripper_dim = (
        load_robomimic_flat(
            offline_data_path, n_demos, abs_action, camera_keys,
            val_ratio=0.02, val_seed=42,
        )
    )

    kwargs = OmegaConf.to_container(cfg.agent, resolve=True)
    kwargs.update(
        seed=cfg.seed,
        num_seeds=cfg.num_seeds,
        updates_per_step=1,
        num_interactions=0,
        observations=env.observation_space.sample()[0, None],
        actions=env.action_space.sample()[0, None],
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        raw_obs_all=raw_obs_all,
        raw_actions_all=raw_actions_all,
        num_robots=num_robots,
        gripper_dim=gripper_dim,
    )
    agent = AgentClass(**kwargs)
    agent.load(cfg.base_checkpoint)
    return agent


# ---------------------------------------------------------------------------
# Offline buffer population
# ---------------------------------------------------------------------------

def populate_offline_buffer(
    base_agent,
    dataset_path: str,
    action_scaler: ActionScaler,
    camera_keys: tuple,
    To: int,
    discount: float,
    obs_keys: tuple,
    max_demos: Optional[int],
    n_step: int = 1,
) -> OfflineResidualBuffer:
    with h5py.File(dataset_path, "r") as f:
        demo_keys = sorted(f["data"].keys())
        if max_demos:
            demo_keys = demo_keys[:max_demos]

        total = sum(
            int(np.argmax(f["data"][d]["dones"][:])) + 1
            for d in demo_keys
        )

        d0       = f["data"][demo_keys[0]]
        obs_dim  = sum(d0["obs"][k].shape[-1] for k in obs_keys)
        img_shape = d0["obs"][camera_keys[0]].shape[1:]
        act_dim  = d0["actions"].shape[-1]

        buffer = OfflineResidualBuffer(
            capacity=total,
            To=To,
            obs_dim=obs_dim,
            act_dim=act_dim,
            image_shape=img_shape,
            camera_keys=tuple(camera_keys),
            discount=discount,
            n_step=n_step,
        )

        for demo_key in tqdm.tqdm(demo_keys, desc="offline buffer", unit="demo"):
            demo = f["data"][demo_key]
            done_idx = int(np.argmax(demo["dones"][:]))
            L = done_idx + 1
            rewards   = demo["rewards"][:L]
            gt_actions = demo["actions"][:L]

            obs_flat_all = np.stack(
                [np.concatenate([demo["obs"][k][t] for k in obs_keys]) for t in range(L)],
                axis=0,
            )  # (L, obs_dim)
            raw_imgs = {cam: demo["obs"][cam][:L] for cam in camera_keys}

            # Run frozen base policy step-by-step (ChunkExecutor tracks its own window)
            base_agent.reset_policy()
            base_actions = np.zeros((L, act_dim), dtype=np.float32)
            for t in range(L):
                raw_base = base_agent.sample_actions(
                    obs_flat_all[t][None],  # (1, obs_dim) — current frame only
                    images={cam: raw_imgs[cam][t][None] for cam in camera_keys},
                )
                base_actions[t] = raw_base[0]

            def _win(arr, t):
                """Build (To, *) window ending at t, padding with frame 0."""
                idxs = [max(t - To + 1 + i, 0) for i in range(To)]
                return arr[idxs]

            for t in range(L - 1):
                obs_win       = _win(obs_flat_all, t)
                obs_win_next  = _win(obs_flat_all, t + 1)
                img_win       = {cam: _win(raw_imgs[cam], t)    for cam in camera_keys}
                img_win_next  = {cam: _win(raw_imgs[cam], t + 1) for cam in camera_keys}

                norm_base      = action_scaler.scale(base_actions[t])
                norm_next_base = action_scaler.scale(base_actions[t + 1])
                norm_combined  = action_scaler.scale(gt_actions[t])

                buffer.add(
                    obs=obs_win,
                    images=img_win,
                    norm_base=norm_base,
                    norm_combined=norm_combined,
                    reward=float(rewards[t]),
                    next_obs=obs_win_next,
                    next_images=img_win_next,
                    next_norm_base=norm_next_base,
                    done=(t == L - 2),
                )

        buffer.finalize()
        print(f"Offline buffer: {len(buffer)} transitions from {len(demo_keys)} demos.")
    return buffer


# ---------------------------------------------------------------------------
# Critic + residual model initialisation
# ---------------------------------------------------------------------------

def _init_critic(obs_flat_dim: int, act_dim: int, cfg: DictConfig, rng):
    from xqc.networks import critic_net

    critic_def = critic_net.VMapCritic(
        hidden_dims=tuple(cfg.critic.hidden_dims),
        n_outputs=cfg.critic.n_bins,
        max_v=cfg.critic.max_v,
        min_v=cfg.critic.min_v,
        n_critics=cfg.critic.n_critics,
        pre_activation_bn=cfg.critic.pre_activation_bn,
        use_layer_norm=False,
        use_batch_norm=cfg.critic.use_batch_norm,
        skip_connections=False,
    )
    dummy_obs    = jnp.zeros((2, obs_flat_dim))
    dummy_action = jnp.zeros((2, act_dim))
    critic = Model.create(
        critic_def,
        inputs=[rng, dummy_obs, dummy_action, True],
        tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(cfg.critic.lr)),
    )
    if cfg.critic.use_weight_norm:
        critic = norm_network(critic)
    return critic, critic   # (live, target) — both start identical


def _init_residual(obs_flat_dim: int, act_dim: int, cfg: DictConfig, rng):
    net = ResidualMLP(
        hidden_dims=tuple(cfg.residual.hidden_dims),
        act_dim=act_dim,
        action_scale=cfg.residual.action_scale,
    )
    dummy_obs  = jnp.zeros((2, obs_flat_dim))
    dummy_base = jnp.zeros((2, act_dim))
    return Model.create(
        net,
        inputs=[rng, dummy_obs, dummy_base],
        tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(cfg.residual.lr)),
    )




# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------

def _linear_schedule(v_max: float, v_min: float, n_steps: int, step: int) -> float:
    """Linear decay from v_max to v_min over n_steps, then held at v_min."""
    mix = float(np.clip(step / max(n_steps, 1), 0.0, 1.0))
    return (1.0 - mix) * v_max + mix * v_min


def _run_critic_warmup(
    critic, target_critic, residual_model,
    online_buffer, offline_buffer,
    base_agent, obs_offset, obs_scale, camera_keys,
    cfg, rng,
):
    """Pure gradient phase: critic_warmup_steps updates on the filled buffers,
    no env stepping, actor stays frozen. Matches residual_td3 reference."""
    for i in tqdm.trange(cfg.critic_warmup_steps, desc="critic warmup", unit="step"):
        batch = sample_mixed(
            online_buffer, offline_buffer,
            cfg.batch_size, cfg.offline_fraction, rng,
        )
        critic, target_critic, c_info = update_critic(
            critic=critic,
            target_critic=target_critic,
            frozen_vision_models=base_agent.vision,
            frozen_obs_encoder_model=base_agent.obs_encoder,
            residual_model=residual_model,
            batch=batch,
            obs_offset=obs_offset,
            obs_scale=obs_scale,
            camera_keys=camera_keys,
            use_weight_norm=cfg.critic.use_weight_norm,
            tau=cfg.critic.tau,
        )
        if (i + 1) % cfg.log_interval == 0:
            wandb.log(
                {**{f"warmup/{k}": float(v)
                    for k, v in c_info.items()
                    if hasattr(v, "ndim") and v.ndim == 0},
                 "warmup_step": i + 1},
            )
    return critic, target_critic


def save_checkpoint(residual_model: Model, critic: Model, directory: Path):
    import orbax.checkpoint as ocp
    directory.mkdir(parents=True, exist_ok=True)
    ckpt = ocp.PyTreeCheckpointer()
    ckpt.save(str(directory / "residual"), {"params": residual_model.params}, force=True)
    ckpt.save(str(directory / "critic"),   {"params": critic.params,
                                             "batch_stats": critic.batch_stats}, force=True)
    print(f"  Saved to {directory}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="residual_config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    wandb.init(
        project=cfg.wandb.project_name,
        entity=cfg.wandb.get("entity"),
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    wandb.define_metric("warmup/*", step_metric="warmup_step")

    rng = np.random.default_rng(cfg.seed)
    jax_rng = jax.random.PRNGKey(cfg.seed)

    camera_keys = tuple(cfg.agent.camera_keys)
    camera_names = [k.removesuffix("_image") for k in camera_keys]
    offline_data_path = cfg.agent.offline_data_path

    from xqc.replay_buffer.offline_replay_buffer import ROBOSUITE_OBS_KEYS
    obs_keys = tuple(k for k in ROBOSUITE_OBS_KEYS if k != "object")

    # --- Environments -------------------------------------------------------
    # Both envs created together before any JAX allocation, mirroring
    # train_parallel_chunking.py — avoids EGL context competition with JAX.
    env = ParallelEnv(
        env_names=[cfg.env.name] * cfg.num_seeds,
        seed=cfg.seed,
        render=True,
        camera_names=camera_names,
        horizon=cfg.env.get("horizon"),
        dataset_path=offline_data_path,
        abs_action=cfg.agent.get("abs_action", False),
        action_repeat=cfg.env.action_repeat,
    )
    eval_env = ParallelEnv(
        env_names=[cfg.env.name] * cfg.eval_num_seeds,
        seed=cfg.seed + 9999,
        render=True,
        camera_names=camera_names,
        horizon=cfg.env.get("horizon"),
        dataset_path=offline_data_path,
        abs_action=cfg.agent.get("abs_action", False),
        action_repeat=cfg.env.action_repeat,
    )

    # --- Base policy -------------------------------------------------------
    print(f"Loading base policy from {cfg.base_checkpoint}...")
    base_agent = _make_base_agent(cfg, env)
    print("  Base policy loaded and frozen.")

    To          = base_agent.To
    act_dim     = base_agent.robot_action_dim  # 7D axis-angle sent to env (not 10D rotation_6d internal)
    obs_offset  = jnp.asarray(base_agent.obs_offset)
    obs_scale   = jnp.asarray(base_agent.obs_scale)
    low_dim_obs_dim = env.observation_space.shape[-1]

    with h5py.File(offline_data_path, "r") as f:
        d0 = f["data"][sorted(f["data"].keys())[0]]
        img_shape = d0["obs"][camera_keys[0]].shape[1:]

    # --- ActionScaler ------------------------------------------------------
    action_scaler = ActionScaler.from_hdf5(
        offline_data_path,
        action_scale=cfg.residual.action_scale,
    )

    # --- Offline buffer ----------------------------------------------------
    offline_buffer = populate_offline_buffer(
        base_agent=base_agent,
        dataset_path=offline_data_path,
        action_scaler=action_scaler,
        camera_keys=camera_keys,
        To=To,
        discount=cfg.discount,
        obs_keys=obs_keys,
        max_demos=cfg.get("max_offline_demos", None),
        n_step=cfg.n_step,
    )

    # --- Probe obs_flat_dim ------------------------------------------------
    dummy_obs  = jnp.zeros((1, To, low_dim_obs_dim))
    dummy_imgs = {cam: jnp.zeros((1, To, *img_shape)) for cam in camera_keys}
    obs_flat_dim = _encode_frozen(
        base_agent.vision, base_agent.obs_encoder,
        dummy_obs, dummy_imgs, obs_offset, obs_scale, camera_keys,
    ).shape[-1]
    print(f"  obs_flat_dim={obs_flat_dim}, act_dim={act_dim}, To={To}")

    # --- Init critic + residual -------------------------------------------
    jax_rng, rng_c, rng_r = jax.random.split(jax_rng, 3)
    critic, target_critic = _init_critic(obs_flat_dim, act_dim, cfg, rng_c)
    residual_model        = _init_residual(obs_flat_dim, act_dim, cfg, rng_r)

    # --- Eval agent (used by eval_env.evaluate — no custom loop needed) ---
    eval_agent = ResidualAgent(
        base_agent=base_agent,
        residual_model=residual_model,
        action_scaler=action_scaler,
        obs_offset=obs_offset,
        obs_scale=obs_scale,
        num_envs=cfg.eval_num_seeds,
        obs_dim=low_dim_obs_dim,
        img_shape=img_shape,
        cfg=cfg,
    )

    # Force JIT compilation of inference functions on dummy data NOW, before
    # any eval env subprocesses are spawned.  Without this, the first eval
    # step triggers a cold-compile of the ResNet vision encoder while EGL
    # renderers are also allocating GPU memory, causing a subprocess crash.
    print("Warming up JIT-compiled inference functions...")
    _wup_obs  = jnp.zeros((1, To, low_dim_obs_dim))
    _wup_imgs = {cam: jnp.zeros((1, To, *img_shape), dtype=jnp.uint8) for cam in camera_keys}
    _wup_flat = jit_encode(base_agent.vision, base_agent.obs_encoder,
                            _wup_obs, _wup_imgs, obs_offset, obs_scale, camera_keys)
    jit_residual(residual_model.params, residual_model.apply_fn,
                  _wup_flat, jnp.zeros((1, act_dim)))
    del _wup_obs, _wup_imgs, _wup_flat
    print("  JIT warm-up done.")

    # --- Online ring buffer -----------------------------------------------
    online_buffer = ResidualReplayBuffer(
        capacity=cfg.online_buffer_size,
        To=To,
        obs_dim=low_dim_obs_dim,
        act_dim=act_dim,
        image_shape=img_shape,
        camera_keys=camera_keys,
        discount=cfg.discount,
        n_step=cfg.n_step,
    )

    # =========================================================================
    # Phase 0: FM-only sanity check — run the base policy directly (no scale/
    # unscale, no RL loop) to confirm it achieves >0% success in the training
    # env.  Prints per-step debug, saves mp4 videos, and reports outcome.
    # Remove once the training loop is confirmed working.
    # =========================================================================
    import cv2 as _cv2
    from hydra.core.hydra_config import HydraConfig as _HydraConfig
    _sanity_video_dir = (
        Path(cfg.save_dir) if cfg.get("save_dir")
        else Path(_HydraConfig.get().runtime.output_dir)
    ) / "sanity_videos"
    _sanity_video_dir.mkdir(parents=True, exist_ok=True)

    _sanity_episodes = 50
    _sanity_successes = 0
    print(f"[sanity] Running {_sanity_episodes} direct FM rollouts in the training env...")
    for _s_ep in range(_sanity_episodes):
        _s_obs   = env.reset()
        _s_imgs  = env.get_images(camera_keys)
        base_agent.reset_policy()
        _s_done = False
        _s_step = 0
        _s_got_goal = False
        _s_frames = []
        # Image and obs diagnostics at episode start
        _img0 = _s_imgs[camera_keys[0]]
        while not _s_done:
            _s_action = base_agent.sample_actions(
                _s_obs, images={cam: _s_imgs[cam] for cam in camera_keys}
            )
            _s_obs, _s_rew, _s_term, _s_trunc, _s_goal = env.step(_s_action)
            _s_imgs = env.get_images(camera_keys)
            _s_done = bool(_s_term[0]) or bool(_s_trunc[0])
            _s_got_goal = _s_got_goal or bool(_s_global_goal := _s_goal[0])
            _s_step += 1
            # Collect render frame for video (every other step to keep file small)
            if _s_step % 2 == 0:
                _s_frame = env.render(num_envs=1)  # (1, H, W, 3)
                _s_frames.append(_s_frame[0])
        _sanity_successes += int(_s_got_goal)
        # Save episode video
        if _s_frames:
            _vid_path = _sanity_video_dir / f"ep{_s_ep}_{'success' if _s_got_goal else 'fail'}.mp4"
            _h, _w = _s_frames[0].shape[:2]
            _writer = _cv2.VideoWriter(str(_vid_path), _cv2.VideoWriter_fourcc(*"mp4v"), 15, (_w, _h))
            for _fr in _s_frames:
                _writer.write(_cv2.cvtColor(_fr, _cv2.COLOR_RGB2BGR))
            _writer.release()
            print(f"  [sanity] video saved → {_vid_path}", flush=True)
    print(f"[sanity] {_sanity_successes}/{_sanity_episodes} episodes succeeded", flush=True)
    print(f"[sanity] Videos saved in {_sanity_video_dir}", flush=True)

    # --- Online rollout setup ---------------------------------------------
    obs         = env.reset()
    images_raw  = env.get_images(camera_keys)
    history     = ObsHistoryWindow(cfg.num_seeds, To, low_dim_obs_dim, img_shape, camera_keys)
    history.reset_all(obs, images_raw)

    base_agent.reset_policy()
    raw_base  = base_agent.sample_actions(obs, images={cam: images_raw[cam] for cam in camera_keys})
    norm_base = np.stack([action_scaler.scale(raw_base[i]) for i in range(cfg.num_seeds)])

    save_dir = Path(cfg.save_dir) if cfg.get("save_dir") else None
    best_success = 0.0
    actor_update_cadence = max(1, cfg.updates_per_step // cfg.actor_updates_per_step)

    def _collect_step(use_actor: bool, norm_base: np.ndarray, stddev: float):
        """One env step. Returns (next_norm_base, rewards, norm_combined, goals, just_finished)."""
        # Action: random noise during Phase 1 buffer warmup, actor + Gaussian noise in Phase 3
        if not use_actor:
            noise_scale = float(cfg.residual.warmup_noise_scale)
            norm_delta = rng.uniform(-noise_scale, noise_scale,
                                     size=norm_base.shape).astype(np.float32)
        else:
            obs_flat = np.array(jit_encode(
                base_agent.vision, base_agent.obs_encoder,
                jnp.asarray(history.obs),
                {cam: jnp.asarray(history.images[cam]) for cam in camera_keys},
                obs_offset, obs_scale, camera_keys,
            ))
            if cfg.get("debug_freeze_residual", False):
                norm_delta = np.zeros_like(norm_base)
            else:
                norm_delta = np.array(jit_residual(
                    residual_model.params, residual_model.apply_fn,
                    jnp.asarray(obs_flat), jnp.asarray(norm_base),
                ))
            if stddev > 0.0 and not cfg.get("debug_freeze_residual", False):
                norm_delta = norm_delta + rng.normal(0.0, stddev,
                                                     size=norm_delta.shape).astype(np.float32)

        norm_combined_ = np.clip(norm_base + norm_delta, -1.0, 1.0)
        raw_combined   = np.stack([action_scaler.unscale(norm_combined_[i])
                                    for i in range(cfg.num_seeds)])

        prev_obs_win  = history.obs
        prev_imgs_win = history.images

        next_obs_, rewards_, dones_, truncs_, goals_ = env.step(raw_combined)
        next_images_raw_ = env.get_images(camera_keys)
        just_finished_   = np.logical_or(dones_, truncs_)

        # Reset only the finished envs in-place; next_obs_ becomes the first
        # obs of the new episode for done envs (mirrors train_parallel_chunking.py).
        # env.reset() (resetting ALL envs) was wrong here.
        if just_finished_.any():
            next_obs_, _, _, _ = env.reset_where_done(next_obs_, dones_, truncs_)
            next_images_raw_ = env.get_images(camera_keys)

        # Shift history window, then fill done envs' entire window with reset obs
        # so the new episode starts with a clean To-frame context.
        history.step(next_obs_, next_images_raw_)
        if just_finished_.any():
            done_ids = np.where(just_finished_)[0]
            history.reset_envs(done_ids, next_obs_, next_images_raw_)

        # Reset base policy for ALL finished envs (done + truncated), then sample
        # next norm_base from the post-reset observation.
        base_agent.reset_policy_idx(just_finished_)
        next_raw_base_ = base_agent.sample_actions(
            next_obs_, images={cam: next_images_raw_[cam] for cam in camera_keys}
        )
        next_norm_base_ = np.stack([action_scaler.scale(next_raw_base_[i])
                                     for i in range(cfg.num_seeds)])

        for i in range(cfg.num_seeds):
            online_buffer.add(
                obs=prev_obs_win[i],
                images={cam: prev_imgs_win[cam][i] for cam in camera_keys},
                norm_base=norm_base[i],
                norm_combined=norm_combined_[i],
                reward=float(rewards_[i]),
                next_obs=history.obs[i],
                next_images={cam: history.images[cam][i] for cam in camera_keys},
                next_norm_base=next_norm_base_[i],
                done=bool(just_finished_[i]),
                env_id=i,
            )

        return next_norm_base_, rewards_, norm_combined_, goals_, just_finished_

    # =========================================================================
    # Phase 1: buffer warmup — fill online buffer, no gradient updates
    # =========================================================================
    _warmup_video_dir = _sanity_video_dir.parent / "warmup_videos"
    _warmup_video_dir.mkdir(parents=True, exist_ok=True)
    _wu_ep, _wu_frames, _wu_got_goal = 0, [], False
    for step in tqdm.trange(1, cfg.learning_starts + 1,
                             desc="buffer warmup", unit="step"):
        norm_base, _, _, goals, just_finished = _collect_step(
            use_actor=False, norm_base=norm_base, stddev=0.0
        )
        if step % 2 == 0:
            _wu_frames.append(env.render(num_envs=1)[0])
        _wu_got_goal = _wu_got_goal or bool(goals[0])
        if just_finished[0] and _wu_frames:
            _vid_path = _warmup_video_dir / f"ep{_wu_ep}_{'success' if _wu_got_goal else 'fail'}.mp4"
            _h, _w = _wu_frames[0].shape[:2]
            _writer = _cv2.VideoWriter(str(_vid_path), _cv2.VideoWriter_fourcc(*"mp4v"), 15, (_w, _h))
            for _fr in _wu_frames:
                _writer.write(_cv2.cvtColor(_fr, _cv2.COLOR_RGB2BGR))
            _writer.release()
            print(f"  [warmup] video saved → {_vid_path}", flush=True)
            _wu_ep += 1
            _wu_frames, _wu_got_goal = [], False

    # =========================================================================
    # Phase 2: critic warmup — pure gradient steps on filled buffers
    # =========================================================================
    if cfg.critic_warmup_steps > 0:
        critic, target_critic = _run_critic_warmup(
            critic, target_critic, residual_model,
            online_buffer, offline_buffer,
            base_agent, obs_offset, obs_scale, camera_keys,
            cfg, rng,
        )

    # =========================================================================
    # Phase 3: main RL loop — env steps + joint critic/actor updates
    # =========================================================================
    c_info: dict = {}
    a_info: dict = {}
    _episode_goal_flags    = np.zeros(cfg.num_seeds, dtype=bool)
    _episode_success_history: list = []
    _episodes_completed    = 0

    _online_video_dir = _sanity_video_dir.parent / "online_videos"
    _online_video_dir.mkdir(parents=True, exist_ok=True)
    _online_frames: list = []

    # Probe: verify scale/unscale roundtrip and print current norm_base stats
    _back_probe = np.stack([action_scaler.unscale(norm_base[i]) for i in range(cfg.num_seeds)])
    _norm2_probe = np.stack([action_scaler.scale(_back_probe[i]) for i in range(cfg.num_seeds)])
    print(f"[probe] norm_base (going into Phase 3) = {norm_base[0]}")
    print(f"[probe] unscale→rescale roundtrip max_diff = {float(np.abs(norm_base - _norm2_probe).max()):.6f}")
    print(f"[probe] ActionScaler lo={action_scaler.lo}  hi={action_scaler.hi}")

    _debug_step_cnt = 0  # counts Phase-3 steps for early-training debug prints
    pbar = tqdm.tqdm(range(1, cfg.total_steps + 1), desc="train", unit="step")
    for step in pbar:
        stddev = _linear_schedule(
            cfg.residual.stddev_max, cfg.residual.stddev_min,
            cfg.residual.stddev_steps, step,
        )
        norm_base, rewards, norm_combined, goals, just_finished = _collect_step(use_actor=True, norm_base=norm_base, stddev=stddev)
        _debug_step_cnt += 1
        if step % 2 == 0:
            _online_frames.append(env.render(num_envs=1)[0])

        # Debug: print every step for the first 30 Phase-3 steps, then every 100 for the first 2000
        _dbg_print = _debug_step_cnt <= 30 or (_debug_step_cnt <= 2000 and _debug_step_cnt % 100 == 0)
        if _dbg_print:
            _raw_sent = np.stack([action_scaler.unscale(norm_combined[i]) for i in range(cfg.num_seeds)])
            print(
                f"[dbg step={_debug_step_cnt}] "
                f"rew={float(rewards[0]):.3f} goal={float(goals[0]):.3f} "
                f"done={bool(just_finished[0])} "
                f"raw_action={np.round(_raw_sent[0], 3)}",
                flush=True,
            )

        # Rolling last-50-episode success rate (mirrors train_parallel_chunking.py)
        _episode_goal_flags |= goals.astype(bool)
        for _env_idx in np.flatnonzero(just_finished):
            _ep_success = float(_episode_goal_flags[_env_idx])
            _episode_success_history.append(_ep_success)
            _episode_goal_flags[_env_idx] = False
            _episodes_completed += 1
            if _online_frames:
                _vid_path = _online_video_dir / f"ep{_episodes_completed:04d}_step{step:07d}_{'success' if _ep_success else 'fail'}.mp4"
                _h, _w = _online_frames[0].shape[:2]
                _writer = _cv2.VideoWriter(str(_vid_path), _cv2.VideoWriter_fourcc(*"mp4v"), 15, (_w, _h))
                for _fr in _online_frames:
                    _writer.write(_cv2.cvtColor(_fr, _cv2.COLOR_RGB2BGR))
                _writer.release()
                _online_frames = []
            if _debug_step_cnt <= 5000:
                print(
                    f"[dbg episode end] step={_debug_step_cnt} ep={_episodes_completed} "
                    f"success={_ep_success:.0f}  history={_episode_success_history}",
                    flush=True,
                )
        _episode_success_history = _episode_success_history[-50:]
        if _episode_success_history:
            wandb.log(
                {
                    "train/success_rate_last50": float(np.mean(_episode_success_history)),
                    "train/episodes_completed": _episodes_completed,
                },
                step=step,
            )

        # Gradient updates (4 critic : 1 actor, matching residual_td3)
        for i in range(cfg.updates_per_step):
            batch = sample_mixed(
                online_buffer, offline_buffer,
                cfg.batch_size, cfg.offline_fraction, rng,
            )
            critic, target_critic, c_info = update_critic(
                critic=critic,
                target_critic=target_critic,
                frozen_vision_models=base_agent.vision,
                frozen_obs_encoder_model=base_agent.obs_encoder,
                residual_model=residual_model,
                batch=batch,
                obs_offset=obs_offset,
                obs_scale=obs_scale,
                camera_keys=camera_keys,
                use_weight_norm=cfg.critic.use_weight_norm,
                tau=cfg.critic.tau,
            )
            # Actor fires only on the last of every actor_update_cadence critic iters
            if (i + 1) % actor_update_cadence == 0 and not cfg.get("debug_freeze_residual", False):
                residual_model, a_info = update_actor(
                    residual_model=residual_model,
                    critic=critic,
                    frozen_vision_models=base_agent.vision,
                    frozen_obs_encoder_model=base_agent.obs_encoder,
                    batch=batch,
                    obs_offset=obs_offset,
                    obs_scale=obs_scale,
                    camera_keys=camera_keys,
                    l2_reg_weight=cfg.residual.l2_reg_weight,
                )

        # Logging
        if step % cfg.log_interval == 0:
            log = {"step": step, "online_buffer_size": len(online_buffer),
                   "train/stddev": stddev}
            log.update({f"critic/{k}": float(v) for k, v in c_info.items()
                        if hasattr(v, "ndim") and v.ndim == 0})
            log.update({f"actor/{k}": float(v) for k, v in a_info.items()
                        if hasattr(v, "ndim") and v.ndim == 0})
            log["train/reward"] = float(rewards.mean())
            wandb.log(log, step=step)
            pbar.set_postfix({
                "r": f"{float(rewards.mean()):.3f}",
                "critic": f"{float(c_info.get('critic_loss', 0)):.3f}" if c_info else "—",
                "σ": f"{stddev:.3f}",
            })

        if step % cfg.eval_interval == 0:
            eval_agent.residual_model = residual_model
            eval_stats = eval_env.evaluate(eval_agent, num_episodes=cfg.eval_episodes)
            success_rate = float(np.mean(eval_stats["goal"]))
            wandb.log({"eval/success_rate": success_rate}, step=step)
            pbar.write(f"[step {step}] success_rate={success_rate:.3f} (best={best_success:.3f})")
            # Reset base ChunkExecutor back to training size after eval.
            base_agent.reset_policy()

            if save_dir and success_rate >= best_success:
                best_success = success_rate
                save_checkpoint(residual_model, critic, save_dir / "best")

        if save_dir and step % cfg.save_interval == 0:
            save_checkpoint(residual_model, critic, save_dir / "latest")

    wandb.finish()


if __name__ == "__main__":
    main()
