#!/usr/bin/env python
import gym
import logging
import multiprocessing
import os.path as osp
import socket
import subprocess
import tensorflow as tf

from baselines import bench, logger
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy

from env.velocity_reward import VelocityRewardEnv

POLICIES = {
    'cnn': CnnPolicy,
    'lstm': LstmPolicy,
    'lnlstm': LnLstmPolicy,
    'mlp': MlpPolicy,
}
def train(num_timesteps, seed, policy, ncpu=None, load=None):
    """Train the osim walker environment using an optimized PPO policy.

    policy can be any of the keys in the above POLICIES dict.

    load is a filename of saved weights, or None.

    """
    ncpu = ncpu or multiprocessing.cpu_count() # default to using all cores
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    gym.logger.setLevel(logging.WARN)
    tf.Session(config=config).__enter__()

    def make_env(rank):
        def env_fn():
            env = VelocityRewardEnv(visualize=False)
            env.seed(seed + 1000 * rank)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            return env
        return env_fn
    nenvs = 2*ncpu
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

LOG_BASE_PATH = '/home/ubuntu/learntorun/results'
PORT_BASE = 8080

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cpus', help='Number of CPUs', type=int, default=1)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=POLICIES.keys(), default='mlp')
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--tb', help='Enable Tensorboard', action='store_true')
    parser.add_argument('--tag', help='Tag for this run in logs', type=str, default=None)
    args = parser.parse_args()

    # possibly start tensorboard
    if args.tb:
        # find the logger dir, possibly as a function of the tag
        logger.LOG_OUTPUT_FORMATS.append('tensorboard')
        dir = osp.join(LOG_BASE_PATH, args.tag) if args.tag else None
        logger.configure(dir=dir)
        dir = osp.join(dir or logger.get_dir(), 'tb')

        # find an open port. note there's a race between this loop and starting tensorboard
        # also note we don't support more than 100 tensorboards running at a time (lol)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for port in range(PORT_BASE, PORT_BASE+100):
            try:
                s.bind(("127.0.0.1", port))
            except socket.error as e:
                # the port is likely already in use; try the next one
                continue
            s.close()
            break

        # start the tensorboard process in the background
        subprocess.Popen(["tensorboard", f"--logdir={dir}", f"--port={port}"])

    train(args.num_timesteps, args.seed, args.policy, ncpu=args.cpus, load=args.load)


if __name__ == '__main__':
    main()
