#!/usr/bin/env python
import os
import os.path as osp
import re
import socket
import subprocess

from baselines import bench, logger

import env
import models

ENVS = {
    'velocity': env.velocity_reward.VelocityRewardEnv,
    'knee': env.knee_reward.KneeRewardEnv,
    'velocity_torso': env.velocity_torso_reward.VelocityTorsoRewardEnv,
}
MODELS = {
    'ppo2_mlp': models.ppo2.ppo2_mlp_defaults,
    'ppo2_mlp_faster_lr': models.ppo2.ppo2_mlp_faster_learning_rate,
    'ppo2_mlp_slower_lr': models.ppo2.ppo2_mlp_slower_learning_rate,
    'ppo2_mlp_smaller_cliprange': models.ppo2.ppo2_mlp_smaller_cliprange,
    'ppo2_mlp_faster_lr_smaller_cliprange': models.ppo2.ppo2_mlp_faster_lr_smaller_cliprange,
}

LOG_BASE_PATH = '/home/ubuntu/learntorun/results'
PORT_BASE = 8080

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cpus', help='Number of CPUs', type=int, default=1)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--model', help='Policy architecture', choices=MODELS.keys(), required=True)
    parser.add_argument('--env', help='Environment to use (defines reward)', choices=ENVS.keys(), required=True)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--tb', help='Enable Tensorboard', action='store_true')
    parser.add_argument('--visualize', help='Enable visualization (slow)', action='store_true')
    parser.add_argument('--tag', help='Tag for this run in logs', type=str, default=None)
    args = parser.parse_args()

    if args.visualize:
        assert not (args.cpus and args.cpus > 1), \
                'parallelization is not supported with visualization'
        args.cpus = 1
        assert not args.tb, 'tensorboard not supported with visualization'

    def make_env(rank):
        def env_fn():
            env = ENVS[args.env](visualize=args.visualize)
            env.seed(args.seed + 1000 * rank)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            return env
        return env_fn

    if not args.load and args.tag:
        # possibly load state from the tag directory if already present
        tag_checkpoints = osp.join(LOG_BASE_PATH, args.tag, 'checkpoints')
        if osp.exists(tag_checkpoints) and os.listdir(tag_checkpoints):
            # load the most recent snapshot
            args.load = osp.join(tag_checkpoints, sorted(os.listdir(tag_checkpoints))[-1])

    if args.tag:
        # ensure that the tag name is unique so that it doesn't collide with current log directories
        VERSIONED_TAG_RE = re.compile('_v[0-9]+$')
        while osp.exists(osp.join(LOG_BASE_PATH, args.tag)):
            if VERSIONED_TAG_RE.search(args.tag):
                # increment the version by one
                args.tag = VERSIONED_TAG_RE.sub(
                        lambda m: '_v' + str(int(m.group(0)[2:])+1), args.tag)
            else:
                # it's not yet versioned - start by appending _v2
                args.tag += '_v2'

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

    MODELS[args.model](args.num_timesteps, args.seed, make_env, ncpu=args.cpus, load=args.load)


if __name__ == '__main__':
    main()
