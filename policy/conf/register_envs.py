import os
import pathlib

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

# env name -> benchmark (xqc/envs/envs.py's _BENCHMARK_REGISTRY). Listed here
# rather than imported from xqc/envs/pick_place.py: that module imports MuJoCo,
# which picks its rendering backend at import time, i.e. before Hydra's env_set
# (MUJOCO_GL=egl, conf/config.yaml) has run.
ENV_BENCHMARKS = {"PickPlaceBottle": "pick_place"}


def resolve_env_default(name, default):
    return os.environ.get(name, default)


def resolve_egl_device_id() -> str:
    """MuJoCo renders on the first GPU in CUDA_VISIBLE_DEVICES (the one JAX uses),
    "0" if unset."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    return cvd.split(",")[0] if cvd else "0"


def resolve_repo_root() -> str:
    """Absolute path to the repository root (this file is policy/conf/register_envs.py).
    Needed for dataset paths in the yaml files because Hydra changes into the
    per-run output directory."""
    return str(pathlib.Path(__file__).resolve().parent.parent.parent)


OmegaConf.register_new_resolver("env_default", resolve_env_default)
OmegaConf.register_new_resolver("egl_device_id", resolve_egl_device_id)
OmegaConf.register_new_resolver("repo_root", resolve_repo_root)

cs = ConfigStore.instance()
for env_name, benchmark in ENV_BENCHMARKS.items():
    cs.store(
        group="env",
        name=env_name,
        node={
            "name": env_name,
            "benchmark": benchmark,
            "action_repeat": 1,
            # null = 1.5 x the longest demo (xqc/envs/pick_place.py)
            "horizon": None,
            # simplified_init/terminate_on_drop (xqc/envs/pick_place.py's
            # own PickPlaceBottleEnv): both default false = today's exact
            # behavior. Override via env.simplified_init=true /
            # env.terminate_on_drop=true on the CLI.
            "simplified_init": False,
            "terminate_on_drop": False,
        },
    )
