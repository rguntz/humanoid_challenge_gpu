import numpy as np
import wandb
from omegaconf import OmegaConf, DictConfig
from termcolor import colored
import flax.traverse_util
import jax


def print_config(cfg: DictConfig):
    """Prints a structured table of hyperparameters."""
    print("\n" + "=" * 70)
    algo_name = cfg.agent.get("name", "AGENT")
    benchmark = cfg.env.get("benchmark", "UNKNOWN")
    env_name = cfg.env.get("name", "UNKNOWN")
    header = f" HYPERPARAMETERS - {algo_name} - {benchmark}/{env_name} "
    print(colored(header.center(70, "="), "cyan", attrs=["bold"]))
    print("=" * 70)

    def print_dict(content, indent=0):
        for k, v in content.items():
            if isinstance(v, (dict, DictConfig)):
                print("  " * indent + colored(f"{k}:", "yellow", attrs=["bold"]))
                print_dict(v, indent + 1)
            else:
                key_str = "  " * indent + colored(f"{k}:", "green")
                # Handle lists or complex types
                if isinstance(v, (list, tuple)):
                    val_str = str(list(v))
                else:
                    val_str = str(v)
                print(f"{key_str:<40} {val_str}")

    # We resolve the config to handle interpolation
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    print_dict(resolved_cfg)
    print("=" * 70 + "\n")


def log_metrics_to_wandb(step, infos, prefix, fps=30):
    """Logs one value per metric under `{prefix}/{info_key}`. Each metric in
    `infos` is an array with one entry per parallel replica (num_seeds
    independent training seeds, or - for robomimic/mimicgen eval via
    SubprocEnvPool - num_seeds parallel eval rollouts of the same policy);
    that array gets averaged into a single number instead of logging seed0,
    seed1, ... individually as separate wandb lines. `renders` is the
    exception: it's video frames, not a scalar, so it isn't averaged - only
    the first (seed 0) replica's video is logged, since eval_video_num_envs
    defaults to 1 anyway."""
    dict_to_log = {}
    for info_key, value in infos.items():
        if info_key == "renders":
            dict_to_log[f"{prefix}/video"] = wandb.Video(value[0], fps=fps, format="mp4")
            print("Logged video of size", value[0].shape)
        else:
            mean_value = float(np.mean(value))
            dict_to_log[f"{prefix}/{info_key}"] = mean_value
            if info_key in ("return", "r", "goal"):
                print(info_key, mean_value)
    wandb.log(dict_to_log, step=step)


def print_total_param_count(models: list):
    total_params = 0

    for model in models:
        params_flat = flax.traverse_util.flatten_dict(model.params)

        for _, value in params_flat.items():
            param_count = 1
            for dim in value.shape:
                param_count *= dim
            total_params += param_count
