#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import gym, logging
from baselines import logger
from env.velocity_reward import VelocityRewardEnv

def train(cpus, num_timesteps, seed):
    from baselines.ppo1 import mlp_policy
    from baselines.trpo_mpi import trpo_mpi
    U.make_session(num_cpu=cpus).__enter__()
    set_global_seeds(seed)
    env = VelocityRewardEnv(visualize=False)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir())
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    trpo_mpi.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=2048,
            max_kl=0.5, cg_iters=10, cg_damping=0.1,
            gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cpus', help='Number of CPUs', type=int, default=1)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    args = parser.parse_args()
    logger.configure()
    train(args.cpus, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
