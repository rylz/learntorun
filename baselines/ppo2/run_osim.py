#!/usr/bin/env python
import gym, logging
from baselines import bench, logger
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
import os.path as osp
import tensorflow as tf
from osim.env import RunEnv

def train(ncpu, num_timesteps, seed, policy):
    """Train the osim walker environment using an optimized PPO policy.

    policy can be one of the following strings:
      - 'cnn'
      - 'lstm'
      - 'lnlstm'

    """
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    gym.logger.setLevel(logging.WARN)
    tf.Session(config=config).__enter__()

    def make_env(rank):
        def env_fn():
            env = RunEnv(visualize=False)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            return env
        return env_fn
    nenvs = 8
    env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
    env = make_env(0)()
    set_global_seeds(seed)
    env = VecFrameStack(env, 4)
    policy = {'cnn' : CnnPolicy, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy}[policy]
    ppo2.learn(policy=policy, env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1))

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cpus', help='Number of CPUs', type=int, default=1)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    args = parser.parse_args()
    logger.configure()
    train(args.cpus, num_timesteps=args.num_timesteps, seed=args.seed, policy=args.policy)


if __name__ == '__main__':
    main()
