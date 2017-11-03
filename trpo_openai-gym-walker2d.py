# Derived from example.py in osim-rl and trpo_mpi/run_mujoco.py
import mujoco_py
from mpi4py import MPI
import gym

import argparse

from baselines.common import set_global_seeds
import os.path as osp
import logging
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines import bench
from baselines.trpo_mpi import trpo_mpi

def train(num_timesteps, seed, visualize=False, model='example'):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    # Load walking environment
    env = gym.make('Walker2d-v1')
    env.reset()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    # set up TRPO policy
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=32, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    # TODO honor visualize flag
    # TODO honor model file/save results
    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', dest='train', action='store_true', default=True)
    parser.add_argument('--test', dest='train', action='store_false', default=True)
    parser.add_argument('--steps', dest='steps', action='store', default=10000, type=int)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
    parser.add_argument('--model', dest='model', action='store', default="example.h5f")
    args = parser.parse_args()
    logger.configure()
    if args.train:
        train(num_timesteps=args.steps, seed=args.seed, visualize=args.visualize, model=args.model)
    else:
        # TODO implement testing
        print('testing not implemented yet')

if __name__ == '__main__':
    main()
