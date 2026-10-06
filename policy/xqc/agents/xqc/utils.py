import jax

from xqc.networks.common import Model


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree.map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params, target_critic.params
    )
    return target_critic.replace(params=new_target_params)
