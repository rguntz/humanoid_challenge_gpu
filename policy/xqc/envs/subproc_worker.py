"""Subprocess-backed parallel env workers, for benchmarks whose underlying
simulator draws its own randomness from Python/numpy's *global* RNG rather
than accepting a per-instance seed (robosuite/robomimic/mimicgen -
make_robomimic_env/make_mimicgen_env take no `seed` argument at all; see
xqc/envs/robomimic.py, xqc/envs/mimicgen.py). Running N such envs side by
side in one process means they'd all draw from one shared, order-dependent
stream - not independent rollouts, and ParallelEnv._get_goal/step both run
sequentially in a Python loop regardless, so there's no wall-clock benefit
either.

Each worker here lives in its own OS process with its own global RNG,
explicitly re-seeded right after the process starts and before its env is
constructed - the standard SubprocVecEnv-style fix (used by e.g.
stable-baselines3) for exactly this class of problem. This gives both
independence (each worker's rollouts are statistically independent) and
speed (physics stepping actually runs concurrently across CPU cores).

JAX/policy inference is NOT done here - it stays centralized on the GPU in
the main process (train_parallel.py calls agent.sample_actions(...) on the
batched observations gathered from all workers, then passes the batched
actions back into step()). Workers only ever do CPU-bound simulator work.
"""
import multiprocessing as mp

import numpy as np


def _get_goal(info: dict):
    if "success" in info:
        return info["success"]
    elif "is_success" in info:
        return info["is_success"]
    elif "solved" in info:
        return info["solved"]
    return 0


def _get_horizon(env):
    from gymnasium.wrappers import TimeLimit

    while True:
        if isinstance(env, TimeLimit):
            return env._max_episode_steps
        if hasattr(env, "horizon"):
            return env.horizon
        if hasattr(env, "env"):
            env = env.env
        else:
            return None


def _worker(remote, parent_remote, env_name, seed, action_repeat, render, camera_names, horizon, dataset_path, abs_action):
    # Deferred import: xqc.envs.envs imports SubprocEnvPool (below) at module
    # level, so importing make_env at module level here would be circular.
    # By the time this function actually runs (in the freshly spawned child),
    # xqc.envs.envs is a fully-formed module - the cycle only exists at
    # import-time ordering, not at call-time.
    import faulthandler, sys
    faulthandler.enable(file=sys.stderr)

    from xqc.envs.envs import make_env

    parent_remote.close()
    # Re-seeding here - inside the child, before env construction - is what
    # actually decouples this worker's RNG stream from every other worker's.
    # Seeding in the parent before spawning would not help: with the "spawn"
    # start method each child re-imports fresh (no inherited state to
    # decouple from), and with "fork" it would actively hurt (children would
    # inherit an identical copy of the parent's RNG state). Explicit
    # per-worker seeding is correct under either start method.
    np.random.seed(seed)
    import random

    random.seed(seed)

    env = make_env(
        env_name, seed=seed, action_repeat=action_repeat, render=render, camera_names=camera_names,
        horizon=horizon, dataset_path=dataset_path, abs_action=abs_action,
    )

    while True:
        cmd, data = remote.recv()
        try:
            if cmd == "step":
                # Printed (and flushed) BEFORE the call, not just on
                # failure: if env.step crashes natively (a segfault inside
                # MuJoCo/robosuite's C code, e.g. from a NaN/out-of-range
                # action reaching the physics solver), the process dies
                # without ever raising a catchable Python exception, so the
                # except block below never runs and never sees it either -
                # this is the only way to see what action was in flight
                # right before such a crash.
                state, reward, terminal, truncate, info = env.step(data)
                remote.send((state, reward, terminal, truncate, _get_goal(info)))
            elif cmd == "reset":
                state, _ = env.reset(seed=data)
                remote.send(state)
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "get_images":
                remote.send(env.get_images(data))
            elif cmd == "spaces":
                remote.send(
                    (
                        env.observation_space.shape,
                        env.observation_space.dtype,
                        env.action_space.shape,
                        env.action_space.dtype,
                        _get_horizon(env),
                    )
                )
            elif cmd == "close":
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown SubprocEnvPool command: {cmd!r}")
        except Exception:
            # Without this, a worker exception (e.g. a bad action produced
            # by an early-training policy driving the physics into a bad
            # state) kills this process with no trace at all - the parent's
            # own remote.recv() then just raises an opaque EOFError, giving
            # no clue what actually happened. Print the real traceback
            # (reaches the job's own stdout/stderr log) before re-raising,
            # so the worker still dies (preserving prior behavior - the
            # parent isn't equipped to keep going with one env silently
            # dropped) but the cause is now visible. cmd/data logged too
            # since the action that triggered a step failure is exactly
            # what's needed to reproduce it.
            import traceback

            print(f"[subproc_worker] worker crashed handling cmd={cmd!r} data={data!r}", flush=True)
            traceback.print_exc()
            raise


class SubprocEnvPool:
    """One env instance per OS process (mirrors the subset of ParallelEnv's
    per-env operations it needs: step/reset/render/get_images/spaces),
    executed concurrently across processes instead of sequentially in a
    Python loop. Long-lived: workers are spawned once in __init__ and reused
    for every step/reset/eval call until close() - env construction
    (loading/compiling the MuJoCo model) is expensive enough that respawning
    per call would eat most of the benefit."""

    def __init__(self, env_names, seed, action_repeat, render, camera_names, horizon=None, dataset_path=None, abs_action=False):
        # "spawn", not the Linux default "fork": forked children inherit a
        # *copy* of the parent's already-advanced global RNG state at fork
        # time, which would silently correlate workers' rollouts (same
        # failure mode this class exists to avoid). Spawn gives each worker a
        # fresh interpreter with no inherited RNG state to decouple from.
        ctx = mp.get_context("spawn")
        self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in env_names])
        self.processes = []
        for i, (work_remote, env_name) in enumerate(zip(work_remotes, env_names)):
            p = ctx.Process(
                target=_worker,
                args=(
                    work_remote, self.remotes[i], env_name, seed + i, action_repeat, render, camera_names, horizon,
                    dataset_path, abs_action,
                ),
                daemon=True,
            )
            p.start()
            self.processes.append(p)
            work_remote.close()

        for remote in self.remotes:
            remote.send(("spaces", None))
        specs = [remote.recv() for remote in self.remotes]
        self.obs_shapes, self.obs_dtypes, self.act_shapes, self.act_dtypes, horizons = zip(*specs)
        self.horizon = horizons[0]

    def step(self, actions):
        """Returns states as a plain list (NOT stacked) - env replicas built
        from different env_names can have different obs dims, and the caller
        (ParallelEnv.step) pads each to a common width before stacking, same
        as the in-process path. rewards/terminals/truncates/goals are always
        scalar per env, so those are safe to stack here."""
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        states = [r[0] for r in results]
        rewards = np.stack([r[1] for r in results])
        terminals = np.stack([r[2] for r in results])
        truncates = np.stack([r[3] for r in results])
        goals = np.stack([r[4] for r in results])
        return states, rewards, terminals, truncates, goals

    def reset_idx(self, idx, seed):
        self.remotes[idx].send(("reset", seed))
        return self.remotes[idx].recv()

    def render(self, num_envs):
        for remote in self.remotes[:num_envs]:
            remote.send(("render", None))
        return np.stack([remote.recv() for remote in self.remotes[:num_envs]])

    def get_images(self, camera_keys):
        for remote in self.remotes:
            remote.send(("get_images", camera_keys))
        per_env = [remote.recv() for remote in self.remotes]
        return {key: np.stack([obs[key] for obs in per_env]) for key in camera_keys}

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
