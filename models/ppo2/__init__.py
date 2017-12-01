from . import common

def ppo2_mlp_defaults(num_timesteps, seed, make_env, ncpu=None, load=None):
    """Wrapper for the common trainer with default hyperparameters using MLP."""
    common.train(num_timesteps, seed, 'mlp', make_env, ncpu=ncpu, load=load)
