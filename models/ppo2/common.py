#!/usr/bin/env python
import gym
import logging
import multiprocessing
import os.path as osp
import tensorflow as tf

from baselines import bench, logger
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy

POLICIES = {
    'cnn': CnnPolicy,
    'lstm': LstmPolicy,
    'lnlstm': LnLstmPolicy,
    'mlp': MlpPolicy,
}

def train(num_timesteps, seed, policy, make_env, ncpu=None, load=None):
    """Train the osim walker environment using an optimized PPO policy.

    policy can be any of the keys in the above POLICIES dict.

    make_env is a function that takes a single integer for process rank and returns a function
    taking no arguments that returns an environment object, e.g.:
        def make_env(rank):
            def env_fn():
                env = VelocityRewardEnv(visualize=False)
                env.seed(seed + rank)
                return env
            return env_fn

    load is a filename of saved weights, or None.

    TODO: expose hyperparameters as kwargs for this function

    """
    ncpu = ncpu or multiprocessing.cpu_count() # default to using all cores
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    gym.logger.setLevel(logging.WARN)
    tf.Session(config=config).__enter__()

    # if there's only one cpu, we also only use one env. otherwise, we found that 2x envs per
    # tf cpu usage works well
    nenvs = 2*ncpu if ncpu > 1 else 1
    env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
    set_global_seeds(seed)
    env = VecFrameStack(env, 4)
    policy = POLICIES[policy]
    ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
               lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
               ent_coef=0.0,
               lr=3e-4,
               cliprange=0.2,
               total_timesteps=num_timesteps,
               save_interval=5, load_path=load)
