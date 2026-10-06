import os
import numpy as np


def get_discount(T, action_repeat=1):
    T /= action_repeat
    discount = np.clip(((T / 5) - 1) / (T / 5), 0.95, 0.995)
    return discount


def is_slurm_job():
    return "SLURM_JOB_ID" in os.environ


def log_slurm_info(wandb_run):
    if is_slurm_job():
        print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}")
        wandb_run.summary["SLURM_JOB_ID"] = os.environ.get("SLURM_JOB_ID")
        wandb_run.summary["SLURM_JOB_NODELIST"] = os.environ.get("SLURM_JOB_NODELIST")


def save_slurm_outputs(wandb_run):
    if is_slurm_job():
        try:
            from subprocess import check_output

            stdout_path = (
                check_output(
                    "scontrol show jobid -d $SLURM_JOB_ID | awk -F= '/StdOut=/{print $2}'",
                    shell=True,
                )
                .decode("utf-8")
                .replace("\n", "")
            )
            stderr_path = (
                check_output(
                    "scontrol show jobid -d $SLURM_JOB_ID | awk -F= '/StdErr=/{print $2}'",
                    shell=True,
                )
                .decode("utf-8")
                .replace("\n", "")
            )
            wandb_run.save(stdout_path)
            wandb_run.save(stderr_path)
        except Exception as e:
            print(f"Failed to save SLURM outputs to wandb: {e}")


def check_hydra_config():
    from hydra.core.hydra_config import HydraConfig

    hydra_cfg = HydraConfig.get()
    for arg in hydra_cfg.overrides.task:
        if arg.startswith("env.name="):
            raise ValueError(
                "Manually setting 'env.name' is not allowed. "
                "Hydra infers settings automatically based on the selected environment. "
                "Please use 'env=<env_name>' to select the environment."
            )
