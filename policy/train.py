import random
import shutil
import sys
from functools import partial
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Local git-submodule dependencies (robosuite, robomimic, mimicgen, myosuite, ...)
# are installed editable via `uv sync`, which registers a MetaPathFinder appended
# to the *end* of sys.meta_path. Each of those submodules is checked out at the
# repo root under a directory with the exact same name as the package itself
# (e.g. ./robosuite/ contains the "robosuite" package) - and since this script's
# own directory (the repo root) sits on sys.path, the default PathFinder resolves
# an (empty) implicit namespace package for that name from the submodule
# directory *before* the editable finder is ever consulted, silently shadowing
# the real package (symptom: `robosuite.make` missing, `pkg.__file__ is None`).
# Moving editable finders to the front of sys.meta_path fixes this for every
# affected submodule at once, without touching sys.path (still needed as-is for
# `conf.register_envs` below, an unrelated repo-root namespace package).
for _finder in list(sys.meta_path):
    # uv/pip's generated editable-install finders are classes with only
    # classmethods, appended to sys.meta_path as the class object itself
    # (not an instance) - so it's `_finder.__name__`, not `_finder.__class__.__name__`.
    if getattr(_finder, "__name__", None) == "_EditableFinder":
        sys.meta_path.remove(_finder)
        sys.meta_path.insert(0, _finder)

import conf.register_envs  # noqa: F401, This is need to register the environments

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    import wandb
    import xqc.utils

    xqc.utils.check_hydra_config()

    # Resume the same wandb run (not a new one) when resuming training:
    # save_dir/wandb_run_id.txt is written once below, right after the first
    # wandb.init() of a fresh run, and read back here on a later resumed run
    # (cfg.resume_from's parent is that same save_dir - see
    # scripts/run_training.sh's own resume_from/resume_step example) so the
    # eval/loss curves continue in
    # place instead of starting a new, disconnected run at step 50001.
    # Falls back to a fresh run if the id file doesn't exist (e.g. resuming
    # a checkpoint saved before this existed).
    resume_run_id = None
    if cfg.get("resume_from"):
        run_id_file = Path(cfg.resume_from).resolve().parent / "wandb_run_id.txt"
        if run_id_file.exists():
            resume_run_id = run_id_file.read_text().strip()

    with wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.agent.get("wandb_project_name") or cfg.wandb.project_name,
        name=cfg.wandb.name or f"{cfg.agent.name}_{cfg.env.name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        # _service_wait=120 (wandb's own default is 30s - explicit request,
        # see conversation history): 4 concurrent seeds per node all
        # importing robosuite/mujoco/JAX and spinning up their own wandb
        # service subprocess at the same moment is a genuine race under
        # heavy node CPU load - hit wandb.sdk.service.service.
        # ServiceStartTimeoutError twice in a row (reproducible, not a
        # one-off flake) on job1/job2's own launches this session. More
        # headroom, not a different mechanism - doesn't change what gets
        # logged, only how long wandb.init() will wait for its own service
        # subprocess before giving up.
        # (wandb >= 0.19: start_method no longer exists, _service_wait is
        # now x_service_wait.)
        settings=wandb.Settings(x_service_wait=120),
        mode=cfg.wandb.mode,
        id=resume_run_id,
        resume="must" if resume_run_id else None,
    ) as wandb_run:
        if cfg.get("save_dir") and not resume_run_id:
            save_dir_path = Path(cfg.save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)
            (save_dir_path / "wandb_run_id.txt").write_text(wandb_run.id)

        env = eval_env = None
        try:
            # Delayed imports to allow Hydra's env_set to take effect
            import h5py
            import jax
            import numpy as np
            import tqdm

            from xqc.agents.actors.bc.bc_learner import BCLearner
            from xqc.agents.actors.dp.dp_learner import DPLearner
            from xqc.agents.actors.fm.fm_learner import FMLearner
            from xqc.replay_buffer.chunked_qc_buffer import Dataset, ReplayBuffer as ChunkedQCReplayBuffer, load_robomimic_flat
            from xqc.envs import ParallelEnv
            from xqc.normalization import RewardNormalizer
            import xqc.logging

            ################################################################################
            # Setup
            ################################################################################

            xqc.utils.log_slurm_info(wandb_run)
            xqc.logging.print_config(cfg)

            np.random.seed(cfg.seed)
            random.seed(cfg.seed)

            # Make Envs
            # camera_names derived from cfg.agent.camera_keys (e.g. "agentview_image"
            # -> "agentview") - available in config before the agent object exists,
            # so envs can be created with the right cameras rendering from the start.
            camera_keys = cfg.agent.get("camera_keys", [])
            camera_names = [key.removesuffix("_image") for key in camera_keys] or None
            make_envs = partial(
                ParallelEnv,
                action_repeat=cfg.env.action_repeat,
                camera_names=camera_names,
                horizon=cfg.env.get("horizon"),
                dataset_path=cfg.agent.get("offline_data_path"),
                abs_action=cfg.agent.get("abs_action", False),
            )
            # render=True whenever camera_keys is set - the online training
            # loop's own rollout needs env.get_images() to work for a vision
            # agent's own sample_actions() call, same as eval_env already
            # gets below (see conversation history: this was previously
            # missing entirely - the online loop never fetched images at
            # all, only eval_env/evaluate() did, so a vision agent's own
            # online RL rollout could never have worked).
            env = make_envs(env_names=[cfg.env.name] * cfg.num_seeds, seed=cfg.seed, render=bool(camera_keys))
            # Camera obs must be captured whenever the agent is vision-based
            # (camera_keys set), independent of eval_video: eval_video only
            # gates whether run_eval's evaluate() call additionally builds a
            # video for wandb upload (train_parallel.py's run_eval, further
            # below) - but ParallelEnv.evaluate() calls get_images()
            # unconditionally for vision agents regardless of that flag
            # (xqc/envs/envs.py's evaluate()), so without this OR, vision
            # agents crash with a KeyError in the env worker at first eval
            # whenever eval_video is left at its default of false.
            #
            # eval_num_seeds (explicit request - see conversation history,
            # and conf/config.yaml's own field docstring): deliberately its
            # OWN env_names list, no longer sharing cfg.num_seeds with the
            # training env above - lets eval run with more parallel
            # replicas (e.g. eval_num_seeds=10, eval_episodes=5 -> 50 total
            # episodes/point) even while online RL forces cfg.num_seeds=1
            # for the training env/replay buffer. Defaults to cfg.num_seeds
            # (see that field's own default), so this is a no-op for every
            # existing script that doesn't set it.
            eval_env = make_envs(
                env_names=[cfg.env.name] * cfg.eval_num_seeds, seed=cfg.seed + 42,
                render=cfg.eval_video or bool(camera_keys),
            )

            agent_kwargs = OmegaConf.to_container(cfg.agent, resolve=True)
            agent_kwargs.update(
                {
                    "seed": cfg.seed,
                    "num_seeds": cfg.num_seeds,
                    "updates_per_step": cfg.updates_per_step,
                    "num_interactions": int(cfg.max_steps / cfg.env.action_repeat),
                    "observations": env.observation_space.sample()[0, None],
                    "actions": env.action_space.sample()[0, None],
                }
            )
            # Every agent here is a ChunkedPolicyLearner (xqc/agents/chunking/
            # chunked_learner.py); DP, FM and BC are ported so far - the other
            # chunked actors (MF/MIP/...) plug in the same way.
            AGENT_REGISTRY = {"DP": DPLearner, "FM": FMLearner, "BC": BCLearner}
            AgentClass = AGENT_REGISTRY[cfg.agent.name]

            # Offline data (xqc/replay_buffer/chunked_qc_buffer.py's
            # load_robomimic_flat) is loaded ONCE here, before the agent
            # exists - train_dataset is handed to the agent directly; the
            # critic-mixing block below reuses it too (via a cheap second
            # Dataset wrapper around the SAME underlying arrays, not a
            # second load - see its own comment for why a literal shared
            # instance would be wrong here).
            # n_demos=null: every demo in the dataset (load_robomimic_flat
            # also uses n_demos as the episode count for the val split).
            n_demos = cfg.agent.n_demos
            if n_demos is None:
                with h5py.File(cfg.agent.offline_data_path, "r") as hf:
                    n_demos = len(hf["data"])
            train_dataset, val_dataset, raw_obs_all, raw_actions_all, num_robots, gripper_dim = load_robomimic_flat(
                cfg.agent.offline_data_path, n_demos, cfg.agent.get("abs_action", False),
                tuple(camera_keys), val_ratio=cfg.val_ratio, val_seed=42,
            )
            agent_kwargs.update({
                "train_dataset": train_dataset, "val_dataset": val_dataset,
                "raw_obs_all": raw_obs_all, "raw_actions_all": raw_actions_all,
                "num_robots": num_robots, "gripper_dim": gripper_dim,
            })

            # critic_name/critic_kwargs (conf/agent/base_agent.yaml's own
            # fields, e.g. critic_name="mlp") pass straight through into
            # ChunkedPolicyLearner.__init__, which builds the critic
            # directly (no *CriticLearner class involved at all - see
            # conversation history). Non-chunked agents (XQCLearner) absorb
            # these two extra keys via their own **kwargs catch-all, same
            # as every other config-only field.
            agent = AgentClass(**agent_kwargs)

            # Epoch-averaged loss (ChunkedPolicyLearner-based agents only,
            # e.g. DP/FM/MF/MIP/MIP-Flow/NF/BC - agent.dataset.size is the
            # windowed offline dataset's own sample count). Logged separately
            # from the per-log_interval point-sample "loss"/"{name}_loss"
            # metric below (train_loop's own agent.update() call) so it's
            # directly comparable to the reference Diffusion Policy repo's
            # own wandb train_loss, which is always an epoch average (one
            # point per epoch, mean over that epoch's ~steps_per_epoch
            # batches), not a per-step point sample - see
            # TrainDiffusionUnetLowdimWorkspace.run's own epoch loop.
            _dataset_size = getattr(getattr(agent, "dataset", None), "size", None)
            steps_per_epoch = max(1, _dataset_size // cfg.agent.batch_size) if _dataset_size else None
            _epoch_loss_sum = 0.0
            _epoch_loss_count = 0
            # Fixed training batch, captured once on the very first update()
            # call below and reused every epoch after - matches the
            # reference's own `train_sampling_batch` (captured once at
            # batch_idx==0 of epoch 0, reused for every later epoch's
            # train_action_mse_error) - see ChunkedPolicyLearner.
            # compute_action_mse's own docstring.
            _train_sampling_batch = None
            # val_action_mse (explicit request - see conversation history):
            # same "fixed batch, captured once, reused" convention as
            # _train_sampling_batch above, but drawn from agent.val_dataset
            # (the held-out split - train_parallel_chunking.py's own
            # offline_buffer, used to mix offline rows into online updates,
            # is built from train_dataset instead, confirmed never touching
            # val_dataset at all - see conversation history) instead of
            # agent.dataset. Reused for compute_action_mse (an action-space
            # MSE, not the abstract training loss) in BOTH the offline
            # epoch_log block below AND the online loop's own logging block
            # - the whole point is one continuous val_action_mse curve
            # spanning the offline->online transition, same bare-key-merge
            # convention as "MSE"/offline_action_mean already use.
            _val_sampling_batch = None

            # Opt-in resume (cfg.resume_from set + agent implements .load()):
            # restores params/opt_state/EMA/normalization stats from a prior
            # save_dir checkpoint. resume_step (must match that checkpoint's
            # step count) is read below to fast-forward the pre-training
            # loop's start index, so already-completed steps aren't redone.
            if cfg.get("resume_from") and hasattr(agent, "load"):
                agent.load(cfg.resume_from)

            # Heuristic discount (xqc.utils.get_discount, episode-length-based)
            # by default - override with a fixed value (e.g. discount=0.99)
            # to match train_critic_sarsa.py's own fixed data.discount
            # config for an apples-to-apples comparison (its SARSA target
            # scale depends heavily on this - see conversation history: for
            # Lift, the heuristic gives 0.95 vs that script's fixed 0.99,
            # which alone explains most of a ~2x gap in q_values/target_q
            # between the two pipelines). Production runs should leave this
            # unset (the heuristic default) unless a fixed value is
            # deliberately preferred after comparing both.
            discount = cfg.get("discount") or xqc.utils.get_discount(
                env.max_episode_steps, cfg.env.action_repeat
            )

            # ChunkedPolicyLearner-based agents (DP/FM/MF/MIP/MIP-Flow/NF/BC)
            # need a windowed (To-step history, h-step action chunk) online
            # buffer, not ParallelReplayBuffer's own flat single-step one - a
            # flat/ring buffer can't safely look BACKWARD across episode
            # boundaries the way an observation-history window needs to (see
            # conversation history for the full design discussion).
            # xqc/replay_buffer/chunked_qc_buffer.py's ReplayBuffer is the
            # online counterpart to that same module's Dataset (used further below)
            # this buffer feeds the critic's own online RL batch - an
            # agent with no critic_name (critic_name=None) never reaches
            # the offline_data_ratio>0/has_critic branch below either, so
            # building this buffer for every chunked agent regardless of
            # critic_name is harmless (unused if there's no critic to feed).
            # num_seeds=1 is only required for ONLINE RL (the "Main XQC
            # training loop" below actually calling replay_buffer.insert()/
            # sample_sequence()) - eval (ParallelEnv.evaluate(),
            # via ChunkExecutor.sample_actions) genuinely supports
            # num_seeds>1 just fine, and conf/agent/eval/default.yaml's
            # own num_seeds=10 default is deliberately an EVAL-parallelism
            # value (matches the reference's own n_test=50 = 10*5 - see
            # that file's own comment), not something wrong to fix there.
            # With action_repeat=1 (always true for robomimic),
            # max_steps=0 makes range(start_step, end_step) below
            # provably empty, so a pure BC-pretraining run (max_steps=0,
            # the common case) never calls the buffer at all - asserting
            # here unconditionally over-restricted every chunked-agent
            # run, including ones that never touch online RL, and as a
            # side effect silently shrank eval from 50 episodes/point
            # down to 5 - see conversation history (caught the hard way).
            if cfg.max_steps > 0:
                assert cfg.num_seeds == 1, (
                    "chunked_qc_buffer.ReplayBuffer (and ChunkedPolicyLearner itself) only support "
                    f"num_seeds=1 for ONLINE RL (max_steps > 0) - got num_seeds={cfg.num_seeds}. "
                    "ChunkedPolicyLearner has no multi-seed/vmap support at all (see conversation history)."
                )
            # ReplayBuffer.create needs an example transition dict up
            # front (a flat, preallocated array per field - see
            # xqc/replay_buffer/chunked_qc_buffer.py's own module
            # docstring) rather than a shape-derived __init__ the old
            # ChunkedParallelReplayBuffer had. act_dim here is the
            # STORED/windowed dim (post abs_action conversion), same
            # derivation that class used internally - see conversation
            # history for why this must NOT be action_space's own raw
            # dim.
            raw_act_dim = env.action_space.shape[-1]
            is_abs_action = cfg.agent.get("abs_action", False)
            # Per-robot raw action dim is 7 (pos3+rot3+gripper1) whether
            # abs or delta - num_robots is only actually used below for
            # abs_action's per-robot axis-angle -> rotation_6d conversion
            # (see _convert_abs_action in chunked_qc_buffer.py), unused
            # but harmless for delta.
            num_robots = raw_act_dim // 7
            act_dim = num_robots * 10 if is_abs_action else raw_act_dim
            example_transition = dict(
                observations=np.zeros(env.observation_space.shape[-1], dtype=np.float32),
                actions=np.zeros(act_dim, dtype=np.float32),
                rewards=np.float32(0.0),
                terminals=np.float32(0.0),
                masks=np.float32(1.0),
            )
            for cam in camera_keys:
                h_, w_ = agent.raw_image_shape[cam]
                example_transition[cam] = np.zeros((h_, w_, 3), dtype=np.uint8)
            replay_buffer = ChunkedQCReplayBuffer.create(example_transition, size=cfg.replay_buffer_size)


            if cfg.agent.reward_normalization:
                reward_normalizer = RewardNormalizer(
                    n_seeds=cfg.num_seeds,
                    gamma=discount,
                )

            # xqc/replay_buffer/chunked_qc_buffer.py's Dataset (offline,
            # loaded once above) replaces OfflineReplayBuffer here entirely
            # - not a second, parallel buffer alongside it (see conversation
            # history). To/h/discount/camera_keys are held as plain local
            # variables here and passed explicitly at each call below - no
            # config-holding wrapper class, matching qc's own Dataset/
            # ReplayBuffer pair exactly (see that module's own docstring).
            # Only built when the agent actually has a critic to feed - the
            # whole point of this batch is to feed the critic step (see
            # chunked_learner.py's own update()); passing one to an agent
            # with no critic_name would have nothing to do with it.
            has_critic = getattr(agent, "critic", None) is not None
            if cfg.agent.offline_data_ratio > 0.0 and has_critic:
                offline_To, offline_h, offline_camera_keys = agent.To, agent.execution_horizon, tuple(cfg.agent.get("camera_keys", ()))
                # A SECOND, cheap Dataset wrapper around the SAME already-
                # loaded arrays (train_dataset, built above) - not
                # train_dataset itself. Both the agent's own actor trainingeffective_lr_total
                # and this critic-mixing path call .sample()/
                # .sample_sequence() independently; Dataset.sample keeps its
                # own epoch-shuffle position (_epoch_order/_epoch_pos) as
                # instance state, so sharing ONE live object between two
                # independent callers would corrupt each other's shuffle
                # progress (see conversation history). Dataset(dict(...))
                # shares the underlying numpy arrays (no second disk read,
                # no copy) while giving this wrapper its own independent
                # bookkeeping.
                offline_buffer = Dataset(dict(train_dataset))

                offline_batch_size = int(cfg.batch_size * cfg.agent.offline_data_ratio)
                # Precompute return statistics for reward normalization
                # (usually done online) - compute_return_stats_from_dataset
                # (xqc/normalization/normalization.py) reads dataset.rewards/
                # dataset.masks directly; a plain Dataset instance supports
                # extra attributes fine (same trick compute_chunked_stats
                # uses), so set them here from the same live per-query
                # sample_sequence the actual training batches use below
                # (To=1/camera_keys=() since only rewards/masks are needed).
                if cfg.agent.reward_normalization:
                    _all_idx = np.arange(offline_buffer.size)
                    _stats_batch = offline_buffer.sample_sequence(_all_idx, sequence_length=offline_h, discount=discount, To=1, camera_keys=())
                    offline_buffer.rewards = _stats_batch.rewards
                    offline_buffer.masks = _stats_batch.masks
                    reward_normalizer.compute_return_stats_from_dataset(dataset=offline_buffer)
            else:
                offline_buffer = None
                offline_batch_size = 0
                reward_normalizer.ignore_offline_stats()
                
            online_batch_size = cfg.batch_size - offline_batch_size
                
            # Define helper functions.
            # Eval metrics (return/goal/completion_speed/video) go under the
            # "eval" wandb section, training-step metrics (loss, etc.) go
            # under "train" - kept apart so they render as separate panels
            # instead of being interleaved under one "offline"/"online"
            # group. `phase` (offline pre-training vs online fine-tuning) is
            # nested inside each as a second segment, e.g. "eval/offline/goal".
            best_eval_return = {"value": float("-inf")}

            def save_eval_videos(renders, step, phase):
                """renders: (num_envs, frames, 3, H, W) uint8 RGB, from ParallelEnv.evaluate.
                One mp4 per rendered env replica, in <save_dir>/eval_videos/step_<step>/
                (next to the checkpoints, never pruned), or in <hydra run
                dir>/eval_videos/step_<step>/ when save_dir is unset."""
                import cv2
                from hydra.core.hydra_config import HydraConfig

                root = Path(cfg.save_dir) if cfg.get("save_dir") else Path(HydraConfig.get().runtime.output_dir)
                video_dir = root / "eval_videos" / f"step_{step}"
                video_dir.mkdir(parents=True, exist_ok=True)
                for env_idx, frames in enumerate(renders):
                    frames = np.transpose(frames, (0, 2, 3, 1))  # (frames, H, W, 3)
                    path = video_dir / f"env{env_idx}.mp4"
                    writer = cv2.VideoWriter(
                        str(path), cv2.VideoWriter_fourcc(*"mp4v"), cfg.get("eval_video_fps", 15),
                        (frames.shape[2], frames.shape[1]),
                    )
                    for frame in frames:
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    writer.release()
                    print(f"Saved eval video {path}")

            def run_eval(step, phase):
                eval_stats = eval_env.evaluate(
                    agent,
                    num_episodes=cfg.eval_episodes,
                    temperature=0.0,
                    render=cfg.eval_video,
                    max_render_steps=cfg.get("eval_video_max_steps", 4000),
                    render_frameskip=cfg.get("eval_video_frameskip", 1),
                    render_num_envs=cfg.get("eval_video_num_envs", 1),
                    render_episodes=cfg.get("eval_video_episodes", 10),
                    verbose=cfg.get("eval_verbose", False),
                )
                # Reset chunk-executor state again once eval is done: eval_env.evaluate()
                # itself calls agent.reset_policy() at ITS OWN start, but the shared
                # ChunkExecutor it drives (chunked_learner.py's self._executor) lazily
                # re-inits _needs_init/_obs_history/_action_buffer/_step_cnt to eval_env's
                # own num_envs (eval_num_seeds) on the first sample_actions call inside
                # eval. If eval_num_seeds != num_seeds (decoupled on purpose - see
                # conf/config.yaml's own eval_num_seeds docstring), that leaves the
                # executor sized for eval_env, not the training env, once eval returns -
                # the training loop's very next reset_policy_idx() call (after its own
                # num_seeds-sized env.step()) then IndexErrors on a boolean-mask size
                # mismatch. Resetting again here forces the training loop's own next
                # sample_actions call to lazily re-init at num_seeds size instead.
                if hasattr(agent, "reset_policy"):
                    agent.reset_policy()
                # Eval videos (eval_video=true): saved as mp4 files in the run's
                # output folder (eval_video_local) and/or uploaded to wandb
                # (eval_video_wandb) - conf/agent/eval/default.yaml.
                if "renders" in eval_stats:
                    if cfg.get("eval_video_local", True):
                        save_eval_videos(eval_stats["renders"], step, phase)
                    if not cfg.get("eval_video_wandb", False):
                        eval_stats.pop("renders")
                log_infos_to_wandb(step, eval_stats, f"eval/{phase}", cfg.get("eval_video_fps", 30))
                # Each key ALSO logged under a bare, phase-agnostic "eval/"
                # prefix (explicit request - see conversation history, and
                # log_metrics/log_val_infos's own matching comments) so e.g.
                # eval/return renders as ONE continuous wandb line across the
                # offline->online transition instead of eval/offline/return
                # and eval/online/return being two separately-named metrics.
                # "renders" (video) skipped, same reasoning as log_metrics.
                eval_scalars = {f"eval/{k}": float(np.mean(v)) for k, v in eval_stats.items() if k != "renders"}
                if eval_scalars:
                    wandb.log(eval_scalars, step=step)

                # DP-only: also log under the literal key the reference
                # Diffusion Policy repo's own eval uses (test/mean_score -
                # diffusion_policy/env_runner/robomimic_image_runner.py's
                # `log_data[prefix+'mean_score']`, prefix='test/'), mirroring
                # our own "goal" metric (task-success indicator - the direct
                # analogue of robomimic's per-episode max_reward for this
                # sparse-reward task). Purely a convenience alias so xqcfd's
                # DP eval curve can be overlaid directly against the
                # reference run's own wandb curve under the same key name -
                # doesn't replace eval/{phase}/goal above, just duplicates it
                # under this name, and only for DP (other agents aren't
                # being compared against this specific reference).
                if cfg.agent.get("name") == "DP":
                    wandb.log({"test/mean_score": float(np.mean(eval_stats["goal"]))}, step=step)

                # Opt-in checkpointing (cfg.save_dir set + agent implements .save()):
                # one checkpoint per eval (named by step), plus a running "best" by
                # eval return.
                if cfg.get("save_dir") and hasattr(agent, "save"):
                    save_root = Path(cfg.save_dir)
                    agent.save(save_root / f"step_{step}")
                    avg_return = float(np.mean(eval_stats["return"]))
                    if avg_return > best_eval_return["value"]:
                        best_eval_return["value"] = avg_return
                        agent.save(save_root / "best")

                    # Opt-in pruning (cfg.checkpoint_keep_last_n set): delete all
                    # but the N most-recent step_<N> checkpoints so a long run
                    # doesn't accumulate one full checkpoint per eval forever.
                    # "best" is never pruned (separate name, not step_-prefixed).
                    # Keeping N>=2 (not just the latest) means a resume can still
                    # fall back one step if the very last checkpoint was left
                    # mid-write by a job that got killed while saving it.
                    # step_<offline_end_step> (the exact offline->online
                    # boundary, i.e. the last purely-BC-pretrained checkpoint)
                    # is likewise exempt - without this, a long online run
                    # silently prunes it away once enough online-phase
                    # checkpoints accumulate, and if online RL later collapses
                    # or diverges there is no way back to a clean pre-RL actor
                    # short of redoing the entire offline pretraining from
                    # scratch - see conversation history.
                    keep_last_n = cfg.get("checkpoint_keep_last_n")
                    if keep_last_n:
                        step_dirs = sorted(
                            (p for p in save_root.iterdir() if p.is_dir() and p.name.startswith("step_")),
                            key=lambda p: int(p.name.split("_")[1]),
                        )
                        for old_dir in step_dirs[:-keep_last_n]:
                            if int(old_dir.name.split("_")[1]) == offline_end_step:
                                continue
                            shutil.rmtree(old_dir)

            def log_metrics(step, infos, phase):
                metrics = xqc.logging.metrics.compute_logging_metrics(agent, infos)
                log_infos_to_wandb(step=step, infos=metrics, video_fps=None, prefix=f"train/{phase}")
                # Every key ALSO logged under a bare, phase-agnostic "train/"
                # prefix (explicit request - see conversation history: not
                # just target_q, every actor/critic metric here) so each one
                # renders as ONE continuous wandb line across the
                # offline->online transition, instead of being split into two
                # separately-named metrics (e.g. train/offline/target_q,
                # train/online/target_q) that wandb would otherwise never
                # connect. "renders" (video) is skipped - log_infos_to_wandb
                # already handles that one specially (wandb.Video, not a
                # scalar), logging it twice would just double-upload the
                # same video for no benefit.
                scalar_metrics = {f"train/{k}": float(np.mean(v)) for k, v in metrics.items() if k != "renders"}
                if scalar_metrics:
                    wandb.log(scalar_metrics, step=step)
                # phase marker (explicit request - see conversation history):
                # 0.0 during offline, 1.0 during online - now that every
                # train/val/eval metric above renders as one continuous line
                # across the offline->online transition, there's no longer
                # any visual cue (like a metric name or color change) marking
                # WHERE that transition actually happened on a given chart.
                # Overlay this on any of those charts to pinpoint it exactly.
                wandb.log({"is_online": 1.0 if phase == "online" else 0.0}, step=step)

            def log_train_infos(step, infos, phase):
                """Logs agent.update()'s raw returned infos as-is (e.g. BCLearner's
                "bc_loss") - cheap, and unlike log_metrics/compute_logging_metrics
                does not assume agent.critic/agent.actor exist, so it's safe for
                every agent. np.atleast_1d because log_metrics_to_wandb expects a
                per-seed array per key, but non-vmapped agents (BCLearner) return
                plain scalars.

                Each agent names its own primary loss "{agent_name}_loss" (e.g.
                "fm_loss", "mf_loss") - correct for reading one run in isolation,
                but it means wandb puts every agent's loss on its own panel (one
                per distinct metric name), so BC/MIP/NF/MF/FM training curves can
                never be visually compared on the same chart. Also log that same
                value under the agent-agnostic key "loss" (kept alongside the
                original, not replacing it) so wandb's default per-metric-name
                panel groups FM/MF/MIP/NF/BC's loss curves together, distinguished
                by run color/name, on train/{phase}/loss.

                train/{phase}/{k} (offline vs online segmented) - restored after
                a brief stint as bare train/{k} (see conversation history: that
                was deliberately to directly compare against
                train_critic_sarsa.py's own flat-key runs; now that comparison is
                done and both pipelines are confirmed to agree, phase segmentation
                is back so a run that transitions offline->online doesn't blend
                both phases' values into one line for any key they share (e.g.
                critic_loss)."""
                infos = dict(infos)
                primary_loss_key = f"{cfg.agent.name.lower()}_loss"
                if primary_loss_key in infos:
                    infos["loss"] = infos[primary_loss_key]
                # weight_information/{phase}/{k}: split OUT of train/{phase}
                # (explicit request from the supervisor - see conversation
                # history) - the per-layer weight_norm/dense_norm/
                # batchnorm_norm(_ratio) family (chunked_learner.py's own
                # _update, split_dense_and_batchnorm_norms/per_layer_norms in
                # xqc/agents/common/chunked_critic_target.py) was drowning the
                # actually-interesting critic_loss/actor_loss/q_values/
                # td_error/success_rate curves under train/{phase} - one
                # xqc_style critic alone logs 4 XQCBlocks x 2 sub-layers x
                # {layer_norm, layer_norm_ratio} + the 6 aggregate weight/
                # dense/batchnorm norm(_ratio) scalars, ~24 panels per phase,
                # ~48 total offline+online. These are pure network-magnitude
                # diagnostics (is a layer's weight scale drifting from init),
                # never something you'd read on the same chart as a loss
                # curve, so they get their own top-level wandb section
                # instead - grad_norm stays under train/{phase} (it's a
                # gradient-magnitude/optimization-health signal, read
                # alongside the loss it belongs to, not a weight-magnitude
                # one - see conversation history if this line needs
                # revisiting).
                def _is_weight_info_key(k):
                    return k in (
                        "weight_norm", "weight_norm_ratio",
                        "dense_norm", "dense_norm_ratio",
                        "batchnorm_norm", "batchnorm_norm_ratio",
                    ) or k.startswith("layer_norm/") or k.startswith("layer_norm_ratio/")

                # actor_weights/{phase}/{k}: the actor's own counterpart of
                # weight_information/{phase} above (chunked_learner.py's own
                # "actor_"-prefixed weight_norm/dense_norm/batchnorm_norm/
                # layer_norm family, added alongside the critic's - see that
                # file's own _update) - kept in ITS OWN section rather than
                # merged into weight_information/{phase} so an agent with
                # both an actor and a critic doesn't get the two networks'
                # same-named panels (e.g. "weight_norm") overwriting each
                # other on the same chart. The "actor_" prefix is stripped
                # here so the panel names match the critic's own exactly
                # (weight_norm, layer_norm/<path>, ...), just under this
                # different top-level section.
                def _is_actor_weight_info_key(k):
                    return k in (
                        "actor_weight_norm", "actor_weight_norm_ratio",
                        "actor_dense_norm", "actor_dense_norm_ratio",
                        "actor_batchnorm_norm", "actor_batchnorm_norm_ratio",
                    ) or k.startswith("actor_layer_norm/") or k.startswith("actor_layer_norm_ratio/")

                weight_infos = {k: v for k, v in infos.items() if _is_weight_info_key(k)}
                actor_weight_infos = {
                    k[len("actor_"):]: v for k, v in infos.items() if _is_actor_weight_info_key(k)
                }
                train_infos = {
                    k: v for k, v in infos.items() if not _is_weight_info_key(k) and not _is_actor_weight_info_key(k)
                }
                log_infos_to_wandb(
                    step=step,
                    infos={k: np.atleast_1d(np.asarray(v)) for k, v in train_infos.items()},
                    video_fps=None,
                    prefix=f"train/{phase}",
                )
                if weight_infos:
                    log_infos_to_wandb(
                        step=step,
                        infos={k: np.atleast_1d(np.asarray(v)) for k, v in weight_infos.items()},
                        video_fps=None,
                        prefix=f"weight_information/{phase}",
                    )
                if actor_weight_infos:
                    # No {phase} segmentation here (unlike train/{phase} and
                    # weight_information/{phase} above) - explicit request:
                    # actor_weights/{k} stays ONE continuous panel across the
                    # offline->online transition (same step axis both loops
                    # already share - i's own range starts where j's own
                    # left off), so a weight/layer norm's actual jump (or
                    # lack of one) at that boundary is visible on one chart
                    # instead of being split across two same-named-but-
                    # different-panel curves.
                    log_infos_to_wandb(
                        step=step,
                        infos={k: np.atleast_1d(np.asarray(v)) for k, v in actor_weight_infos.items()},
                        video_fps=None,
                        prefix="actor_weights",
                    )

            def log_val_infos(step, infos, phase):
                """Critic-side val diagnostics (agent.compute_critic_val_loss) -
                val/{phase}/{k}, matching log_train_infos's own phase
                segmentation above (only ever called with phase='offline'
                today - compute_critic_val_loss is only wired into the offline
                loop - but takes phase explicitly rather than hardcoding it, so
                an eventual online-phase call site doesn't silently collide
                with the offline curve under the same key).

                Each key ALSO logged under a bare, phase-agnostic "val/"
                prefix (explicit request - see conversation history, and
                log_metrics's own matching comment above) so it renders as ONE
                continuous wandb line across the offline->online transition
                once an online call site exists, instead of two separately-
                named metrics wandb would never connect."""
                atleast1d_infos = {k: np.atleast_1d(np.asarray(v)) for k, v in infos.items()}
                log_infos_to_wandb(step=step, infos=atleast1d_infos, video_fps=None, prefix=f"val/{phase}")
                scalar_infos = {f"val/{k}": float(np.mean(v)) for k, v in atleast1d_infos.items()}
                if scalar_infos:
                    wandb.log(scalar_infos, step=step)

            def log_infos_to_wandb(step, infos, prefix, video_fps=None):
                xqc.logging.log_metrics_to_wandb(
                    step,
                    infos,
                    prefix,
                    video_fps,
                )

            def log_hessian_to_wandb(step):
                xqc.logging.log_metrics_to_wandb(
                        step,
                        xqc.logging.metrics.compute_hessian_eigenspectrum(
                            agent, replay_buffer, cfg.batch_size, cfg.num_seeds
                        ),
                        "hessian",
                    )

            ################################################################################
            # XQC pre-training loop
            ################################################################################

            # critic_pre_training_steps (conf/critic/base_critic.yaml, lives
            # inside critic_kwargs) can be smaller than the actor's own
            # actor_pre_training_steps (conf/actor/base_actor.yaml, the loop
            # bound below) - once j passes it, the critic simply stops
            # getting fed a batch (freezes) while the actor keeps BC
            # training - see conversation history. Defaults to the actor's
            # own count (i.e. "never freeze early") if unset, matching the
            # old single-shared-value behavior.
            critic_pre_training_steps = cfg.agent.critic_kwargs.get(
                "critic_pre_training_steps", cfg.agent.actor_pre_training_steps
            )
            # The loop's own range must cover WHICHEVER of the two counts is
            # larger - if it were bounded by actor_pre_training_steps alone,
            # resuming with actor_pre_training_steps=0 (already-done actor,
            # letting the critic catch up to a larger critic_pre_training_steps)
            # would produce an EMPTY range and the critic would silently get
            # zero additional steps despite its own count not being reached
            # yet - see conversation history (caught before it shipped).
            offline_end_step = max(cfg.agent.actor_pre_training_steps, critic_pre_training_steps)
            for j in tqdm.tqdm(
                range(cfg.get("resume_step", 0) + 1, offline_end_step + 1),
                smoothing=0.1,
                disable=xqc.utils.is_slurm_job(),
            ):
                # Independent per-step gates: the actor stops updating past
                # its own actor_pre_training_steps (update_actor=False below
                # - see ChunkedPolicyLearner.update()'s own docstring) while
                # the critic stops getting fed a batch past ITS OWN
                # (possibly different) critic_pre_training_steps - either
                # can finish first and freeze while the other keeps going,
                # the loop itself keeps running until BOTH are done.
                update_actor_this_step = j <= cfg.agent.actor_pre_training_steps
                # offline_buffer is None exactly when the agent has no
                # critic to feed (see its own construction above) - the
                # whole point of this batch is update_critic, so there's
                # nothing to sample when there's no critic. Also frozen
                # (treated as if there were no critic batch) once j exceeds
                # critic_pre_training_steps, even though offline_buffer
                # itself is still alive - see this loop's own setup comment
                # above.
                # Kept in sync with the num_updates= passed to agent.update()
                # below - ChunkedPolicyLearner.update()'s own critic step
                # expects `batch` pre-stacked num_updates deep along axis 0
                # (one full batch per update iteration), matching
                # xqc_learner.py's own _do_multiple_updates convention (see
                # chunked_learner.py's own update() docstring).
                critic_num_updates = 1
                if offline_buffer is not None and j <= critic_pre_training_steps:
                    # Dataset.sample_parallel_multibatch: critic_num_updates
                    # separate batches, stacked along a leading axis -
                    # chunked agents are single-seed (no num_seeds axis to
                    # batch over like OfflineReplayBuffer's own - see
                    # conversation history), so this is the shape
                    # ChunkedPolicyLearner.update() expects either way.
                    offline_batch = offline_buffer.sample_parallel_multibatch(
                        online_batch_size + offline_batch_size, critic_num_updates,
                        To=offline_To, h=offline_h, gamma=discount, camera_keys=offline_camera_keys,
                    )
                    # Was unconditional (reward_normalizer.normalize(...)
                    # called regardless of cfg.agent.reward_normalization) -
                    # a real pre-existing bug: reward_normalizer itself is
                    # only ever constructed inside `if
                    # cfg.agent.reward_normalization:` above, so setting
                    # that flag False would NameError here rather than
                    # actually disabling normalization, unlike the online
                    # loop's own identical step further below (which always
                    # had this guard) - see conversation history (found
                    # while explaining why this pipeline's critic curves
                    # differ from train_critic_sarsa.py's own, which never
                    # normalizes rewards at all).
                    if cfg.agent.reward_normalization:
                        normalized_rewards = reward_normalizer.normalize(offline_batch.rewards)
                        offline_batch = offline_batch._replace(rewards=normalized_rewards)
                else:
                    offline_batch = None

                infos = agent.update(
                    offline_batch, num_updates=critic_num_updates, pre_train=True,
                    update_actor=update_actor_this_step,
                )

                # Epoch-averaged loss/lr/val_loss/train_action_mse_error -
                # see this loop's own steps_per_epoch/_epoch_loss_sum/
                # _train_sampling_batch setup comments above. All logged as
                # bare keys (no prefix), matching the reference Diffusion
                # Policy repo's own wandb metric names exactly
                # (TrainDiffusionUnetLowdimWorkspace.run's own step_log -
                # 'train_loss'/'lr'/'val_loss'/'train_action_mse_error') -
                # not run through log_infos_to_wandb/log_metrics_to_wandb,
                # which would prefix them (e.g. "train/offline/loss_epoch_avg")
                # and put them on different wandb panels than the
                # reference's own curves, defeating the point of a directly-
                # overlaid comparison.
                if steps_per_epoch is not None:
                    if _train_sampling_batch is None and hasattr(agent, "dataset"):
                        # Matches chunked_learner.py's own real training draw
                        # exactly (Ta-widened, same seeded _dataset_rng) -
                        # see conversation history: Dataset.sample now needs
                        # To/h/gamma explicitly (gamma is an unused RL-only
                        # field for this IL diagnostic batch).
                        _train_sampling_batch = agent.dataset.sample(
                            agent.batch_size, To=agent.To, h=agent.execution_horizon, gamma=1.0,
                            camera_keys=agent.camera_keys, rng=agent._dataset_rng, Ta=agent.Ta,
                        )
                    if _val_sampling_batch is None and hasattr(agent, "val_dataset") and agent.val_dataset.size > 0:
                        # Fixed held-out batch, same "captured once" convention
                        # as _train_sampling_batch just above - get_val_batch
                        # (unlike Dataset.sample) draws every held-out window
                        # with no shuffling, capped to agent.batch_size here
                        # (matching _train_sampling_batch's own size) rather
                        # than the whole held-out set, so compute_action_mse
                        # below stays a cheap, single forward pass each call -
                        # see conversation history.
                        _val_sampling_batch = agent.val_dataset.get_val_batch(
                            To=agent.To, h=agent.execution_horizon, gamma=1.0,
                            camera_keys=agent.camera_keys, Ta=agent.Ta, max_size=agent.batch_size,
                        )
                    _primary_loss_key = f"{cfg.agent.name.lower()}_loss"
                    if _primary_loss_key in infos:
                        _epoch_loss_sum += float(np.asarray(infos[_primary_loss_key]).mean())
                        _epoch_loss_count += 1
                    if j % steps_per_epoch == 0 and _epoch_loss_count > 0:
                        epoch_log = {"train_loss": _epoch_loss_sum / _epoch_loss_count}
                        if hasattr(agent, "lr_schedule_fn"):
                            # agent.state (one combined TrainState) no longer exists post-
                            # refactor (xqc/agents/chunking/chunked_learner.py: three separate
                            # Models - network/vision/obs_encoder - each with their own step
                            # counter). agent.network.step is safe here: this branch only
                            # runs for ChunkedPolicyLearner-based agents (XQCLearner has no
                            # lr_schedule_fn attribute at all), and all three Models' .step
                            # always advance together by construction (one joint gradient
                            # step per agent.update() call - see actor.py's update_actor_bc).
                            epoch_log["lr"] = float(agent.lr_schedule_fn(agent.network.step))
                        if hasattr(agent, "compute_val_loss"):
                            val_loss = agent.compute_val_loss()
                            if val_loss is not None:
                                epoch_log["val_loss"] = val_loss
                        if hasattr(agent, "compute_action_mse") and _train_sampling_batch is not None:
                            # ChunkedBatch has many more fields than the
                            # former 3-field ChunkBatch (observations/
                            # actions/images) - explicit attribute access,
                            # not positional unpacking (see conversation
                            # history).
                            _obs, _act, _imgs = _train_sampling_batch.observations, _train_sampling_batch.actions, _train_sampling_batch.images
                            epoch_log["train_action_mse_error"] = agent.compute_action_mse(_obs, _act, _imgs)
                            if _val_sampling_batch is not None:
                                # val_action_mse (explicit request - see
                                # conversation history): same
                                # compute_action_mse call as
                                # train_action_mse_error just above, fed the
                                # held-out _val_sampling_batch instead of the
                                # training one - a literal action-space MSE
                                # against demonstrations the actor never
                                # trains on, not just the abstract training
                                # objective (val_loss above already covers
                                # that). Bare key, logged the same way as
                                # train_action_mse_error - the online loop's
                                # own logging block below reuses this exact
                                # key so it forms one continuous curve across
                                # the offline->online transition.
                                epoch_log["val_action_mse"] = agent.compute_action_mse(
                                    _val_sampling_batch.observations, _val_sampling_batch.actions, _val_sampling_batch.images,
                                )
                            if hasattr(agent, "compute_action_range"):
                                # offline_action_{mean,max,min} (explicit
                                # request - see conversation history): the
                                # offline-phase counterpart of update_actor's
                                # own online-phase offline_action_{mean,max,min}/
                                # online_action_{mean,max,min}
                                # (xqc/agents/chunking/actor.py) - bare keys,
                                # logged directly via this block's own
                                # wandb.log(epoch_log, ...) call below (same
                                # "MSE" bare-key-merge convention just below),
                                # so this reads as one continuous curve
                                # spanning offline BC pretraining straight
                                # into online RL.
                                _action_range = agent.compute_action_range(_obs, _imgs)
                                epoch_log["offline_action_mean"] = _action_range["mean"]
                                epoch_log["offline_action_max"] = _action_range["max"]
                                epoch_log["offline_action_min"] = _action_range["min"]
                            # "MSE" (explicit request - see conversation
                            # history): a bare, phase-agnostic key combining
                            # this (offline BC-vs-demonstration MSE) with the
                            # online phase's own "offline_prior_mse"
                            # (live-actor-vs-frozen-BC-prior MSE, logged
                            # separately below) under one shared wandb metric
                            # name - the two never coincide in step (this one
                            # only exists during the offline loop, that one
                            # only once online), so on one wandb chart this
                            # reads as a single continuous curve: BC fit
                            # quality while offline, then drift-from-prior
                            # once online. Kept ALONGSIDE the original
                            # train_action_mse_error, not replacing it.
                            epoch_log["MSE"] = epoch_log["train_action_mse_error"]
                        # Clearly-namespaced duplicates of the bare DP-reference
                        # keys just above (train_loss/lr/val_loss/
                        # train_action_mse_error - kept as-is for direct overlay
                        # against the reference Diffusion Policy repo's own
                        # wandb curves, see this block's own comment above) -
                        # those bare names are ambiguous once you're looking at
                        # more than one agent/phase at a time ("whose loss?
                        # offline or online?" - see conversation history), so
                        # every agent/phase now also gets an unambiguous
                        # actor/offline/{...} copy.
                        for _bare, _clear in (
                            ("train_loss", "actor/offline/loss"),
                            ("lr", "actor/offline/lr"),
                            ("val_loss", "actor/offline/val_loss"),
                            ("train_action_mse_error", "actor/offline/action_mse"),
                        ):
                            if _bare in epoch_log:
                                epoch_log[_clear] = epoch_log[_bare]
                        if hasattr(agent, "critic_lr_schedule_fn"):
                            # ChunkedPolicyLearner-only (critic_lr_schedule_fn
                            # is the critic's own warmup_cosine_decay_schedule,
                            # same shape as the actor's own lr_schedule_fn -
                            # see chunked_learner.py's own critic_tx
                            # construction and conversation history).
                            # agent.critic.step is safe here for the same
                            # reason agent.network.step is safe for the
                            # actor's own "lr" just above - no
                            # reset_optimizers() has fired yet during the
                            # offline phase, so it hasn't gone stale (see
                            # this file's own online-phase actor/online/lr
                            # comment below for why that stops being true
                            # once online RL starts). XQCLearner
                            # (xqc/xqcfd/crossq) has no critic_lr_schedule_fn
                            # at all - hasattr keeps this safe for both agent
                            # families.
                            epoch_log["critic/offline/lr"] = float(agent.critic_lr_schedule_fn(agent.critic.step))
                            # "critic_lr" (bare, explicit request - see
                            # conversation history): never had a
                            # phase-agnostic form before (unlike the actor's
                            # own "lr" above) - added now so it's continuous
                            # across the offline->online transition from the
                            # start, matching "lr"'s own convention.
                            epoch_log["critic_lr"] = epoch_log["critic/offline/lr"]
                        wandb.log(epoch_log, step=j * cfg.env.action_repeat)
                        _epoch_loss_sum = 0.0
                        _epoch_loss_count = 0

                # Policy Evaluation
                if (cfg.get("eval_at_start", True) and j == 1) or j % cfg.eval_interval == 0:
                    run_eval(step=j * cfg.env.action_repeat, phase='offline')


                # Logging
                if j == 1 or j % cfg.log_interval == 0:
                    log_train_infos(step=j * cfg.env.action_repeat, infos=infos, phase='offline')
                    if cfg.agent.log_gradient_metrics and j > cfg.start_training_critic:
                        log_metrics(step=j * cfg.env.action_repeat, infos=infos, phase='offline')
                    # Critic val diagnostics (held-out set, no gradient) -
                    # same cadence as the train logging just above, matching
                    # train_critic_sarsa.py's own val/{k} keys for direct
                    # comparability (see conversation history). Only while
                    # the critic is actually being fed a batch this step
                    # (mirrors offline_batch's own gating above - matches
                    # the domain train/critic_loss itself is meaningful
                    # over) - hasattr guards every non-ChunkedPolicyLearner
                    # agent (e.g. XQCLearner has no compute_critic_val_loss).
                    if hasattr(agent, "compute_critic_val_loss") and offline_batch is not None:
                        val_info = agent.compute_critic_val_loss(discount)
                        if val_info is not None:
                            log_val_infos(step=j * cfg.env.action_repeat, infos=val_info, phase='offline')


            # Only do this the FIRST time execution transitions from
            # offline to online - i.e. resume_step is still within (or
            # exactly at) the offline phase. If resume_step is already
            # PAST offline_end_step, a PRIOR invocation already made
            # this exact transition once (already snapshotted/reset
            # then) - re-doing it here on every resume would overwrite
            # prior_network with the already-drifted, currently-fine-
            # tuning actor (losing the original BC baseline the whole
            # mechanism is supposed to anchor against) AND wipe out
            # already-accumulated Adam momentum every time the job gets
            # resubmitted mid-online-RL (e.g. across multiple 24h
            # resubmissions of a long online run on Isambard) - see
            # conversation history. Doesn't apply to the XQCfD branch
            # above - a separate, pre-existing code path not touched
            # here.
            first_online_transition = cfg.get("resume_step", 0) <= offline_end_step
            if first_online_transition:
                # Save the prior (BC-pretrained) policy for future
                # online-RL KL regularization - same "freeze the actor
                # right before online RL begins" moment as XQCfD's own
                # prior_actor snapshot above, just three Models
                # (network/vision/obs_encoder) instead of XQCfD's one
                # bundled `actor` - ChunkedPolicyLearner.__init__ builds
                # prior_network/prior_vision/prior_obs_encoder for
                # exactly this (tx=None, never gradient-updated
                # directly). Not read by anything yet - see
                # conversation history: scaffolding for later, this
                # class doesn't use KL regularization yet.
                agent.prior_network = agent.prior_network.replace(
                    params=agent.network.params, batch_stats=agent.network.batch_stats
                )
                agent.prior_vision = {
                    cam: agent.prior_vision[cam].replace(params=m.params, batch_stats=m.batch_stats)
                    for cam, m in agent.vision.items()
                }
                agent.prior_obs_encoder = agent.prior_obs_encoder.replace(
                    params=agent.obs_encoder.params, batch_stats=agent.obs_encoder.batch_stats
                )
                # ChunkedPolicyLearner's own actor optimizer was built
                # with lr_schedule=cosine_warmup, decay_steps=
                # actor_pre_training_steps (ChunkedPolicyLearner.__init__)
                # - the schedule's own step counter lives inside
                # opt_state (optax's internal count, not the separate
                # Model.step bookkeeping field), so by the time the
                # offline loop above finishes, the actor's effective LR
                # has already decayed to (and would stay at) 0 for the
                # rest of training. reset_optimizers() re-zeroes
                # opt_state (via tx.init(params), keeping the already-
                # trained params) for both actor and critic, restarting
                # the schedule fresh for online fine-tuning - same fix
                # XQCfD already applies at this exact transition point,
                # for the same underlying reason. See conversation
                # history.
                #
                # online_steps: actor_pre_training_steps/
                # critic_pre_training_steps are the OFFLINE budget -
                # reusing that same decay_steps for the freshly-reset
                # online schedule would decay over the wrong span (see
                # conversation history for the concrete numeric
                # example). cfg.start_training_critic/start_training_actor
                # aren't subtracted out here either (this schedule budget
                # is still sized off the FULL online window, not online
                # window minus either rollout-only delay computed below
                # as `start_training_{critic,actor} = cfg.start_training_
                # {critic,actor} + start_step`) - a minor overestimate of
                # how many updates will actually happen (real critic
                # updates only begin once i > start_training_critic,
                # actor updates only once i > start_training_actor - see
                # conversation history: this delay used to be a silent
                # no-op for a chunked agent, since i already starts at
                # offline_end_step+1, past cfg.start_training's old raw
                # value - now genuinely respected), not something that's
                # broken, just not perfectly calibrated. max(1,
                # ...) keeps this well-defined even for a max_steps=0
                # diagnostic run (online loop never actually runs, so
                # this schedule is simply never queried in that case).
                online_update_budget = max(1, cfg.max_steps // cfg.env.action_repeat) * cfg.updates_per_step
                agent.reset_optimizers(online_steps=online_update_budget)

            ################################################################################
            # Main XQC training loop
            ################################################################################

            # offline_end_step (not actor_pre_training_steps alone) - the
            # offline loop above actually ran up to max(actor, critic)
            # pre_training_steps, so the online loop must resume from
            # THERE, whichever of the two was larger, to avoid either
            # re-running or skipping steps (see conversation history).
            #
            # max(..., resume_step + 1): resume_step can ALSO already be
            # past offline_end_step - i.e. resuming mid-ONLINE-RL, not just
            # mid-offline (a previous invocation already ran some online
            # steps before this job got requeued/resubmitted). Without this,
            # start_step would always reset to offline_end_step+1 regardless
            # of resume_step, silently RE-RUNNING every online step already
            # completed in that earlier invocation - see conversation
            # history. end_step is unaffected: it's the absolute target
            # step (offline_end_step + max_steps), same convention as
            # actor_pre_training_steps/critic_pre_training_steps - it
            # doesn't shift just because resume_step moved start_step up.
            start_step = max(1 + offline_end_step, cfg.get("resume_step", 0) + 1)
            end_step = (cfg.max_steps + offline_end_step) // cfg.env.action_repeat + 1
            # cfg.start_training_critic/cfg.start_training_actor (each
            # default 5_000, i.e. equal - reproduces the old single
            # cfg.start_training's behavior exactly when neither is
            # overridden) are each meant as "how many ONLINE rollout steps
            # to take before that component's first real update" - not an
            # absolute step count. Without this offset it was silently a
            # no-op for a chunked agent: i already starts at start_step
            # (>= offline_end_step+1, e.g. 50001 or 100001), already past
            # the raw config value, so real updates began on the very FIRST
            # online step regardless of what cfg.start_training said -
            # confirmed this session to be the direct cause of a
            # near-instant policy collapse right at the offline->online
            # transition (compounded by this same session's LR schedule
            # change removing the online phase's own warmup ramp - see
            # conversation history's full derivation and the
            # debug_mf_xqc_fix_delayed_updates diagnostic that verified
            # this fix's effect empirically before it was made permanent
            # here). Adding start_step makes it a genuine, online-relative
            # rollout-only warmup again.
            #
            # Split into two independent thresholds (explicit request - see
            # conversation history): lets a run keep the CRITIC fine-tuning
            # online from close to the offline->online transition while
            # deliberately delaying the ACTOR's own first online gradient
            # step much later (chunked agents only - ChunkedPolicyLearner.
            # update()'s own update_actor=False skips just the actor's
            # gradient step, matching the exact mechanism already used to
            # decouple actor_pre_training_steps/critic_pre_training_steps
            # during OFFLINE pretraining, reused here for the online phase).
            # Purpose: isolate whether the critic's own online fine-tuning
            # regime is a smooth continuation of its offline training (as
            # you'd expect at ~90% success, where most new online states
            # should still be roughly in-distribution) or a genuinely
            # different regime - with a frozen (not-yet-updating) actor
            # still doing the rollout during that window, any regime change
            # observed in the critic's own metrics can't be blamed on the
            # actor moving underneath it at the same time.
            #
            # start_training_actor must be >= start_training_critic (the
            # actor updating before the critic has had ANY chance to adapt
            # online would defeat the entire point) - checked directly on
            # the raw cfg values (order is preserved either way, since both
            # get the identical +start_step offset next) so a misconfigured
            # run fails immediately at startup rather than well into a long
            # job.
            assert cfg.start_training_actor >= cfg.start_training_critic, (
                f"start_training_actor ({cfg.start_training_actor}) must be >= "
                f"start_training_critic ({cfg.start_training_critic}) - the actor should never start "
                "online fine-tuning before the critic has had at least as much chance to adapt online first."
            )
            # rollout_warmup_steps (explicit request - see conversation
            # history and conf/config.yaml's own field docstring): the
            # first rollout_warmup_steps online iterations run the policy
            # and step the env exactly as normal (action sampling, env.step,
            # success_rate_last50 tracking - all already unconditional,
            # untouched below) but do NOT insert into replay_buffer (see
            # insert_start_step's own use further below) and do NOT take any
            # gradient step (enforced here by flooring BOTH
            # start_training_critic/start_training_actor's own +start_step
            # offset at rollout_warmup_steps too - a gradient step before
            # insertion has ever happened would sample from an empty
            # buffer). Purpose: let success_rate_last50 actually fill up
            # with real post-offline-pretraining rollout data (not still
            # thinned out by the first <50 episodes right at the
            # transition) before fine-tuning starts changing the policy
            # underneath it. Default 0 (conf/config.yaml) reproduces
            # today's behavior exactly - insert_start_step==start_step,
            # no warmup window at all.
            insert_start_step = start_step + cfg.get("rollout_warmup_steps", 0)
            start_training_critic = max(cfg.start_training_critic, cfg.get("rollout_warmup_steps", 0)) + start_step
            start_training_actor = max(cfg.start_training_actor, cfg.get("rollout_warmup_steps", 0)) + start_step
            update_count = 0
            observations = env.reset()
            # Rolling last-50-completed-episode success rate, computed
            # directly from this training rollout's own env.step() goal
            # signal - NOT from the separate eval_env's periodic
            # evaluate() calls (see conversation history: this answers "is
            # the rollout itself succeeding", independent of eval).
            # _episode_goal_flags tracks, per parallel env, whether the
            # goal has fired at least once in the CURRENT (still-running)
            # episode - same "did it ever succeed" convention
            # ParallelEnv.evaluate() uses for its own goals accumulator.
            # Global across envs (not per-env) since we only care about the
            # last 30 completed episodes overall, not per-replica.
            _online_num_envs = observations.shape[0]
            _episode_goal_flags = np.zeros(_online_num_envs, dtype=bool)
            _episode_success_history = []
            _episodes_completed = 0  # cumulative, unlike _episode_success_history (capped at 30)

            for i in tqdm.tqdm(
                range(start_step, end_step),
                smoothing=0.1,
                disable=xqc.utils.is_slurm_job(),
            ):
                # Sample actions - ChunkedPolicyLearner has no log-prob action
                # sampling at all (a generative/deterministic chunked actor,
                # unlike XQC's own stochastic policy - see conversation
                # history: sample_actions_with_log_probs unconditionally
                # raises NotImplementedError there), so chunked agents go
                # through the receding-horizon sample_actions executor
                # instead - same one ParallelEnv.evaluate() already uses.
                # images fetched ONCE per step (not once for sample_actions
                # and again for replay_buffer.insert() below) and reused for
                # both - env was constructed with render=bool(camera_keys)
                # above specifically so this has frames to fetch.
                images = env.get_images(agent.camera_keys) if getattr(agent, "camera_keys", None) else None
                # No random-action warmup (start_training's own
                # replay-buffer-fill / first-gradient-step delay below is
                # untouched, still real) - unlike typical from-scratch
                # off-policy RL, the actor here is already BC-pretrained
                # before online RL starts, so it's not random noise at
                # step 1. Sampling env.action_space instead of the policy
                # during warmup would push a competent policy into
                # out-of-distribution actions it would never actually take,
                # filling the replay buffer with unrepresentative
                # transitions - sample from the policy from the first step
                # instead. See conversation history.
                if images is not None:
                    actions = agent.sample_actions(observations, images=images)
                else:
                    actions = agent.sample_actions(observations)

                # Step env
                next_observations, rewards, dones, truncs, step_goals = env.step(actions)
                if cfg.agent.reward_normalization:
                    reward_normalizer.update(rewards, np.logical_or(dones, truncs))

                # Update + log the rolling last-50-episode success rate (see
                # this loop's own setup comment above) - every step, per
                # explicit request, not just when an episode happens to end
                # this step (reflects whatever the window's current mean
                # is, even mid-episode). Before any episode has completed
                # yet, there's nothing meaningful to log, so skip.
                _episode_goal_flags = _episode_goal_flags | step_goals.astype(bool)
                for _env_idx in np.flatnonzero(np.logical_or(dones, truncs)):
                    _episode_success_history.append(float(_episode_goal_flags[_env_idx]))
                    _episode_goal_flags[_env_idx] = False
                    _episodes_completed += 1
                _episode_success_history = _episode_success_history[-50:]
                if _episode_success_history:
                    wandb.log(
                        {
                            "train/online/success_rate_last50": float(np.mean(_episode_success_history)),
                            "train/online/episodes_completed": _episodes_completed,
                        },
                        step=i * cfg.env.action_repeat,
                    )

                # Save to buffer - skipped during the rollout_warmup_steps
                # window (see insert_start_step's own docstring above):
                # env.step() and success_rate_last50 tracking above still
                # happen every iteration regardless, only insertion (and,
                # via start_training_critic/start_training_actor's own
                # matching floor above, gradient updates) are held back.
                if i >= insert_start_step:
                    masks = env.generate_masks(dones, truncs)
                    replay_buffer.insert(
                        observations, actions, rewards, masks, truncs, next_observations, images=images,
                        abs_action=is_abs_action, num_robots=num_robots,
                    )
                observations = next_observations

                # Captured BEFORE reset_where_done - that call mutates its
                # own truncs argument in place, zeroing it for any env that
                # just finished (xqc/envs/envs.py's reset_where_done: `states[j],
                # terminals[j], truncates[j] = self._reset_idx(j), False, False`).
                # Reading dones/truncs AFTER that call (as this used to)
                # silently dropped every truncation-ended episode from
                # reset_policy_idx below - ParallelEnv.evaluate() gets this
                # right by computing its own equivalent BEFORE calling
                # reset_where_done (see conversation history for the full
                # bug - a persistent, compounding ChunkExecutor staleness
                # for any episode that ends via truncation, not just the
                # first one).
                just_finished = np.logical_or(dones, truncs)

                # Reset env
                observations, terms, truncs, _ = env.reset_where_done(
                    observations, dones, truncs
                )
                # Reset the chunked actor's own receding-horizon executor
                # state (ChunkExecutor's _obs_history/_image_history/
                # _action_buffer/_step_cnt) for whichever envs just reset -
                # same call ParallelEnv.evaluate() already makes after its
                # own reset_where_done. Without this, sample_actions() keeps
                # SHIFTING the previous (now-ended) episode's own stale
                # observation history into the new episode's early steps
                # instead of starting fresh - confirmed directly this
                # session (a standalone eval script missing this call
                # measured 0.25 success on a checkpoint whose own real
                # training-time eval was 0.92 - see conversation history).
                if hasattr(agent, "reset_policy_idx"):
                    agent.reset_policy_idx(just_finished)

                # Update agent - gated on the CRITIC's own (earlier-or-equal)
                # threshold: the critic always updates first once its own
                # warmup has passed, matching the pre-existing
                # actor_pre_training_steps/critic_pre_training_steps
                # decoupling already used during OFFLINE pretraining
                # (ChunkedPolicyLearner.update()'s own update_actor=False
                # skips just the actor's gradient step below, reused here
                # for the same purpose online - see this loop's own
                # start_training_critic/start_training_actor comment above).
                if i > start_training_critic:
                    # Same two-step pattern qc's own main_online.py uses
                    # inline (not a buffer method there either - see
                    # xqc/replay_buffer/chunked_qc_buffer.py's own module
                    # docstring): draw one big flat batch of indices,
                    # call sample_sequence, then reshape into
                    # (updates_per_step, online_batch_size, ...) - the
                    # leading-axis shape ChunkedPolicyLearner.update()
                    # expects (one full batch per update iteration).
                    _idx = np.random.randint(replay_buffer.size, size=(cfg.updates_per_step * online_batch_size,))
                    _flat_batch = replay_buffer.sample_sequence(
                        _idx, sequence_length=agent.execution_horizon, discount=discount, To=agent.To,
                        camera_keys=camera_keys, mask_ta=agent.Ta,
                    )
                    batches = jax.tree.map(
                        lambda x: x.reshape(cfg.updates_per_step, online_batch_size, *x.shape[1:]), _flat_batch
                    )
                    # offline_mask: (updates_per_step, batch) - 1.0 for a row
                    # drawn from offline_buffer, 0.0 for online - only ever
                    # built/used by the wasserstein_offline_only masking path
                    # (xqc/agents/chunking/actor.py's own update_actor), None
                    # otherwise (agent.update()'s own default - harmless for
                    # every agent that isn't using that flag). Built here
                    # rather than as a ChunkedBatch field since which buffer
                    # a row came from is already known trivially at THIS
                    # concatenation site (online_batch_size/offline_batch_size
                    # are plain ints right here) - see conversation history.
                    offline_mask = None
                    if cfg.agent.offline_data_ratio > 0.0:
                        # ReplayBuffer (online) is a Dataset subclass
                        # (chunked_qc_buffer.py) - same ChunkedBatch
                        # return type as offline_buffer here, so mixing
                        # is just "sample the offline side the same way
                        # via Dataset's own sample_parallel_multibatch,
                        # then concatenate along the batch axis" - no
                        # bespoke ChunkedBatch.concat() needed (an
                        # earlier version of this comment claimed that
                        # was a missing prerequisite; it isn't -
                        # jax.tree.map already concatenates NamedTuples
                        # generically). See conversation history.
                        offline_batches = offline_buffer.sample_parallel_multibatch(
                            offline_batch_size, cfg.updates_per_step,
                            To=agent.To, h=agent.execution_horizon, gamma=discount, camera_keys=camera_keys,
                        )
                        # axis=1: axis 0 is the updates_per_step leading
                        # axis (see batches' own reshape above), axis 1
                        # is the actual per-update batch dimension -
                        # concatenating there grows online_batch_size +
                        # offline_batch_size = cfg.batch_size samples per
                        # update, matching what ChunkedPolicyLearner.update()
                        # expects either way (pure-online or mixed).
                        batches = jax.tree.map(lambda a, b: np.concatenate([a, b], axis=1), batches, offline_batches)
                        offline_mask = np.concatenate(
                            [
                                np.zeros((cfg.updates_per_step, online_batch_size), dtype=np.float32),
                                np.ones((cfg.updates_per_step, offline_batch_size), dtype=np.float32),
                            ],
                            axis=1,
                        )

                    if cfg.agent.reward_normalization:
                        normalized_rewards = reward_normalizer.normalize(
                            batches.rewards
                        )
                        batches = batches._replace(rewards=normalized_rewards)

                    # debug_skip_agent_update: rollout/buffer-insert/sample
                    # and eval all still run normally, only the actual
                    # gradient update is skipped - lets a run validate that
                    # env stepping + resumed-checkpoint action sampling
                    # reproduce the checkpoint's own performance in
                    # isolation from whether the update path is correct
                    # (see conversation history).
                    if not cfg.get("debug_skip_agent_update", False):
                        # offline_mask/update_actor: ChunkedPolicyLearner.
                        # update()-only kwargs (offline_mask is
                        # wasserstein_offline_only's own masking input - see
                        # actor.py's own update_actor docstring; update_actor
                        # is this loop's own start_training_actor gate) -
                        # XQCLearner.update() has neither parameter, so both
                        # must stay out of the call entirely for a
                        # non-chunked agent rather than being passed through.
                        # update_actor=(i > start_training_actor): the critic
                        # already updates unconditionally once this whole
                        # block runs (i > start_training_critic, checked
                        # above) - this additionally skips just the actor's
                        # own gradient step (identity pass-through, see
                        # ChunkedPolicyLearner.update()'s own docstring)
                        # until the actor's own, later-or-equal threshold is
                        # reached, leaving a frozen (rollout-only) actor
                        # during that window while the critic fine-tunes
                        # alone.
                        update_kwargs = {"offline_mask": offline_mask, "update_actor": i > start_training_actor}
                        infos = agent.update(batches, num_updates=cfg.updates_per_step, pre_train=False, **update_kwargs)
                        update_count += cfg.updates_per_step

                ################################################################################
                # Evaluation and Logging
                ################################################################################

                # Policy Evaluation
                if (cfg.get("eval_at_start", True) and i == 1) or i % cfg.eval_interval == 0:
                    run_eval(step=i * cfg.env.action_repeat, phase='online')

                # Logging (infos only exists when the update above actually
                # ran - gated on start_training_critic, matching the update
                # block's own gate above, NOT start_training_actor: infos
                # still exists and is worth logging even while the actor's
                # own gradient step is being skipped, since it carries the
                # critic-only update's own critic_loss/td_error/etc.)
                if (
                    i > start_training_critic
                    and i % cfg.log_interval == 0
                    and not cfg.get("debug_skip_agent_update", False)
                ):
                    log_train_infos(step=i * cfg.env.action_repeat, infos=infos, phase='online')
                    # "MSE" (explicit request - see the offline phase's own
                    # identical-purpose comment, train_action_mse_error's
                    # own epoch_log block above): a bare, phase-agnostic
                    # key sharing one wandb metric name with that offline-
                    # only one, so the two overlay as a single continuous
                    # curve across the whole run. Direct wandb.log (not
                    # through log_infos_to_wandb, which would prefix this
                    # to "train/online/MSE" - a DIFFERENT key from the
                    # offline side's bare "MSE", defeating the whole point).
                    # "offline_prior_mse" only exists in infos when
                    # actor.py's own update_actor actually computed it
                    # (wasserstein_regularization on, actor_name=="MFLearner",
                    # offline_mask has offline rows that step - see its own
                    # docstring) - omitted here too when absent, matching
                    # that key's own "nothing meaningful to report" convention.
                    if "offline_prior_mse" in infos:
                        wandb.log({"MSE": float(np.asarray(infos["offline_prior_mse"]).mean())}, step=i * cfg.env.action_repeat)
                    # offline_action_{mean,max,min}/online_action_{mean,max,min}
                    # (explicit request - see conversation history): same
                    # bare-key-merge convention as "MSE" just above - direct
                    # wandb.log (not through log_train_infos, which would
                    # prefix these to "train/online/offline_action_mean" -
                    # a DIFFERENT key from the offline epoch_log's own bare
                    # "offline_action_mean", defeating the whole point) so
                    # offline_action_{mean,max,min} overlays as ONE
                    # continuous curve across offline BC pretraining and
                    # online RL (ChunkedPolicyLearner.compute_action_range's
                    # own offline-phase counterpart, train_parallel_chunking.py's
                    # own epoch_log block above), while online_action_{mean,
                    # max,min} only ever appears once online (no offline
                    # equivalent - there's no online buffer yet during BC).
                    # Only ever in infos when actor.py's own update_actor
                    # actually computed it (offline_mask not None - see its
                    # own docstring) - omitted here too when absent.
                    if "offline_action_mean" in infos:
                        wandb.log(
                            {
                                "offline_action_mean": float(np.asarray(infos["offline_action_mean"])),
                                "offline_action_max": float(np.asarray(infos["offline_action_max"])),
                                "offline_action_min": float(np.asarray(infos["offline_action_min"])),
                                "online_action_mean": float(np.asarray(infos["online_action_mean"])),
                                "online_action_max": float(np.asarray(infos["online_action_max"])),
                                "online_action_min": float(np.asarray(infos["online_action_min"])),
                            },
                            step=i * cfg.env.action_repeat,
                        )
                    # val_loss/val_action_mse continuing into online RL
                    # (explicit request - see conversation history): both
                    # already exist offline (this loop's own epoch_log
                    # block above) - same bare keys, same "MSE"/
                    # offline_action_mean bare-key-merge convention, so both
                    # form one continuous curve spanning the offline->online
                    # transition instead of stopping dead at the transition
                    # like every other offline-only diagnostic. Reuses
                    # _val_sampling_batch (a FIXED batch, captured once
                    # during the offline loop above - see its own
                    # declaration) rather than redrawing one every call.
                    # compute_val_loss has no such fixed-batch need (it
                    # already iterates its own held-out set internally each
                    # call - see its own docstring) - called fresh each time
                    # here, matching how it's called offline.
                    if hasattr(agent, "compute_val_loss"):
                        _online_val_loss = agent.compute_val_loss()
                        if _online_val_loss is not None:
                            wandb.log({"val_loss": _online_val_loss}, step=i * cfg.env.action_repeat)
                    if hasattr(agent, "compute_action_mse") and _val_sampling_batch is not None:
                        wandb.log(
                            {
                                "val_action_mse": agent.compute_action_mse(
                                    _val_sampling_batch.observations, _val_sampling_batch.actions, _val_sampling_batch.images,
                                )
                            },
                            step=i * cfg.env.action_repeat,
                        )
                    if cfg.agent.log_gradient_metrics:
                        log_metrics(step=i * cfg.env.action_repeat, infos=infos, phase='online')
                    # actor/online/lr: queried with update_count (gradient
                    # updates taken since the online phase's own optimizer
                    # reset - see this loop's own update_count/reset_freq
                    # bookkeeping above), NOT agent.network.step. Those two
                    # counters only agree before the offline->online
                    # transition's reset_optimizers() call: that resets the
                    # LR schedule's own internal counter (living in
                    # opt_state) back to 0, restarting a fresh
                    # warmup_steps=500 + decay_steps=actor_pre_training_steps
                    # cycle for online fine-tuning, but leaves
                    # agent.network.step (a separate, never-reset bookkeeping
                    # counter - see chunked_learner.py's own reset_optimizers
                    # docstring) at its stale pre-reset value - querying the
                    # schedule with THAT would log a wrong LR that doesn't
                    # match what's actually being applied. See conversation
                    # history: whether re-running a full
                    # actor_pre_training_steps-long decay cycle during online
                    # RL (rather than e.g. a shorter online-specific one) is
                    # itself a separate, still-open design question - this
                    # only makes the logged value ACCURATE to whatever the
                    # schedule actually does, not a judgment on that.
                    if hasattr(agent, "lr_schedule_fn"):
                        # "lr" (bare, explicit request - see conversation
                        # history): the offline loop's own epoch_log["lr"]
                        # (above) is the phase-agnostic/DP-reference-matching
                        # key this is meant to continue under - without this,
                        # "lr" silently stops at the offline->online
                        # transition (only the namespaced actor/online/lr
                        # continues), hiding exactly the LR-reset behavior
                        # (jump back to peak, cosine->linear) that's most
                        # important to see continuously across that boundary.
                        wandb.log(
                            {"actor/online/lr": float(agent.lr_schedule_fn(update_count)), "lr": float(agent.lr_schedule_fn(update_count))},
                            step=i * cfg.env.action_repeat,
                        )
                    if hasattr(agent, "critic_lr_schedule_fn"):
                        # Same update_count reasoning as actor/online/lr just
                        # above - reset_optimizers() resets the critic's own
                        # opt_state (hence its schedule's internal counter)
                        # at the exact same offline->online transition as the
                        # actor's, and one agent.update() call advances both
                        # by the same amount, so update_count is equally
                        # correct here. See the offline block's own identical
                        # hasattr comment - ChunkedPolicyLearner-only.
                        wandb.log(
                            {"critic/online/lr": float(agent.critic_lr_schedule_fn(update_count)), "critic_lr": float(agent.critic_lr_schedule_fn(update_count))},
                            step=i * cfg.env.action_repeat,
                        )

                # Condition number logging (expensive) - start_training_critic
                # (not start_training_actor): marks when updates first begin
                # at all, same as the update block's own gate above.
                if cfg.log_interval_condition_number and (
                    i == start_training_critic
                    or i % cfg.log_interval_condition_number == 0
                ):
                    log_hessian_to_wandb()

                # Reset agent
                if cfg.agent.reset_freq and update_count >= cfg.agent.reset_freq:
                    agent.reset()
                    update_count = 0

        finally:
            xqc.utils.save_slurm_outputs(wandb_run)
            # No-op for the in-process backend; tears down SubprocEnvPool
            # workers (robomimic/mimicgen with num_seeds>1) if present, so
            # they don't linger after the run ends or errors out. env/eval_env
            # may still be None if setup failed before they were constructed.
            if env is not None:
                env.close()
            if eval_env is not None:
                eval_env.close()


if __name__ == "__main__":
    main()
