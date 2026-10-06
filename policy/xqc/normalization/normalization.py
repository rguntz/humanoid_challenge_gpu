import flax.linen.initializers as initializers
import numpy as np

default_kernel_init = initializers.lecun_normal()


class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(
        self, epsilon=1e-8, shape=(), dtype=np.float64, max_v=-np.inf, min_v=np.inf
    ):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, dtype=dtype)
        self.mean_squared = np.zeros(shape, dtype=dtype)
        self.var = np.ones(shape, dtype=dtype)
        self.max = np.ones(shape, dtype=dtype) * -np.inf
        self.min = np.ones(shape, dtype=dtype) * np.inf
        self.count = np.zeros(shape, dtype=dtype)
        self.epsilon = epsilon

    def update(self, x, idxs=None):
        """Updates the mean, var and count from a batch of samples."""
        # if the batch size is 1, as the rewards that are collected from the env, it is often collapsed
        if x.shape == self.mean.shape:
            x = x[None]
        # update all seeds
        if idxs is None:
            idxs = np.arange(self.mean.shape[0])

        batch_mean = np.mean(x, axis=0)
        batch_mean_squared = np.mean(np.square(x), axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(
            batch_mean, batch_mean_squared, batch_var, batch_count, idxs
        )
        self.max[idxs] = np.maximum(self.max, abs(x.squeeze()))[idxs]
        self.min[idxs] = np.minimum(self.min, abs(x.squeeze()))[idxs]

    def update_from_moments(
        self, batch_mean, batch_mean_sq, batch_var, batch_count, idxs
    ):
        """Updates from batch mean, variance and count moments."""
        self.mean[idxs], self.mean_squared[idxs], self.var[idxs], self.count[idxs] = (
            update_mean_var_count_from_moments(
                self.mean[idxs],
                self.mean_squared[idxs],
                self.var[idxs],
                self.count[idxs],
                batch_mean[idxs],
                batch_mean_sq[idxs],
                batch_var[idxs],
                batch_count,
            )
        )


def update_mean_var_count_from_moments(
    mean, mean_sq, var, count, batch_mean, batch_mean_sq, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    delta_sq = batch_mean_sq - mean_sq

    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    new_mean_sq = mean_sq + delta_sq * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_mean_sq, new_var, new_count


def normalize(observations, rms, epsilon=1e-5):
    mean, var = rms.mean, rms.var
    to_expand = observations.ndim - mean.ndim
    new_shape = (observations.shape[0],) + (1,) * to_expand + (mean.shape[-1],)
    mean, var = mean.reshape(new_shape), var.reshape(new_shape)
    return (observations - mean) / np.sqrt(var + epsilon)


class RewardNormalizer:
    def __init__(self, n_seeds, gamma):
        self.gamma = gamma
        self.G = np.zeros(n_seeds)
        self.online_G_rms = RunningMeanStd(shape=(n_seeds,))
        self.offline_G_rms = RunningMeanStd(shape=(n_seeds,))
        self.offline_G_var_max = np.full(shape=(n_seeds,), fill_value=-np.inf)

        self.epsilon = 1e-8

    def update(self, reward, done):
        self.G = self.gamma * (1 - done) * self.G + reward
        self.online_G_rms.update(self.G)

    def normalize(self, rewards, **kwargs):
        if self.offline_G_var_max is not None:
            denum = np.maximum(np.sqrt(self.online_G_rms.var), np.sqrt(self.offline_G_var_max))
        else:
            denum = np.sqrt(self.online_G_rms.var)
        # Was a hardcoded np.expand_dims(denum, axis=(1, 2)) - assumes
        # rewards is always exactly 3D (XQCLearner's own (n_seeds, ...,
        # ...) convention). denum itself is always 1D (shape (n_seeds,) -
        # see __init__), so expand dynamically to however many trailing
        # dims rewards actually has instead - a strict generalization
        # (reproduces the old axis=(1,2) behavior exactly when rewards is
        # 3D) that also correctly handles ChunkedPolicyLearner's own 2D
        # (num_updates, batch_size) rewards (n_seeds=1 there always - see
        # conversation history: the old hardcoded version silently
        # broadcast a 2D rewards array into an extra phantom leading axis,
        # (num_updates, batch_size) -> (1, num_updates, batch_size),
        # corrupting only the rewards/actor_entropy fields of a
        # per-update-sliced ChunkedBatch and crashing the online RL
        # critic's TD-target arithmetic).
        to_expand = rewards.ndim - denum.ndim
        denum = denum.reshape(denum.shape + (1,) * to_expand)
        return rewards / (denum + self.epsilon)
    
    def ignore_offline_stats(self):
        self.offline_G_var_max = None
    
    def compute_return_stats_from_dataset(self, dataset):
        def update_offline(reward, done):
            self.G = self.gamma * (1 - done) * self.G + reward
            self.offline_G_rms.update(self.G)
            self.offline_G_var_max = np.maximum(self.offline_G_var_max, self.offline_G_rms.var)

        rewards = dataset.rewards
        episode_over = 1 - dataset.masks
        for i in range(rewards.shape[0]):
            update_offline(reward=rewards[i], done=episode_over[i])
