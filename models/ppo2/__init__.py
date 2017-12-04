from . import common

def ppo2_mlp_defaults(num_timesteps, seed, make_env, ncpu=None, load=None):
    """Wrapper for the common trainer with default hyperparameters using MLP."""
    common.train(num_timesteps, seed, 'mlp', make_env, ncpu=ncpu, load=load)


def ppo2_mlp_faster_learning_rate(num_timesteps, seed, make_env, ncpu=None, load=None):
    """PPO2 with MLP and a 3.3x faster than default learning rate."""
    common.train(num_timesteps, seed, 'mlp', make_env, ncpu=ncpu, load=load, lr=1e-3)


def ppo2_mlp_slower_learning_rate(num_timesteps, seed, make_env, ncpu=None, load=None):
    """PPO2 with MLP and a 3x slower than default learning rate."""
    common.train(num_timesteps, seed, 'mlp', make_env, ncpu=ncpu, load=load, lr=1e-4)
