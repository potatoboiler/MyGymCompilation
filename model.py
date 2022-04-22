"""
Need to fetch datasets from server
"""
import math
import random
from collections import deque, namedtuple
from itertools import count

import compiler_gym
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from compiler_gym.wrappers import (ConstrainedCommandline, CycleOverBenchmarks,
                                   TimeLimit)
from PIL import Image
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

import clientIO


def make_env() -> compiler_gym.envs.CompilerEnv:
    """Make the reinforcement learning environment for this experiment."""
    # We will use LLVM as our base environment. Here we specify the observation
    # space from this paper: https://arxiv.org/pdf/2003.00671.pdf and the total
    # IR instruction count as our reward space, normalized against the
    # performance of LLVM's -Oz policy.
    env = compiler_gym.make(
        "llvm-v0",
        observation_space="Autophase",
        reward_space="IrInstructionCountOz",
    )
    '''
    # Here we constrain the action space of the environment to use only a 
    # handful of command line flags from the full set. We do this to speed up
    # learning by pruning the action space by hand. This also limits the 
    # potential improvements that the agent can achieve compared to using the 
    # full action space.
    env = ConstrainedCommandline(env, flags=[
        "-break-crit-edges",
        "-early-cse-memssa",
        "-gvn-hoist",
        "-gvn",
        "-instcombine",
        "-instsimplify",
        "-jump-threading",
        "-loop-reduce",
        "-loop-rotate",
        "-loop-versioning",
        "-mem2reg",
        "-newgvn",
        "-reg2mem",
        "-simplifycfg",
        "-sroa",
    ])
    '''
    # Finally, we impose a time limit on the environment so that every episode
    # for 5 steps or fewer. This is because the environment's task is continuous
    # and no action is guaranteed to result in a terminal state. Adding a time
    # limit means we don't have to worry about learning when an agent should
    # stop, though again this limits the potential improvements that the agent
    # can achieve compared to using an unbounded maximum episode length.
    env = TimeLimit(env, max_episode_steps=5)
    return env




def train():
    if ray.is_initialized():
        ray.shutdown()
    ray.init(include_dashboard=False, ignore_reinit_error=True)

    train_benchmarks = lambda x : print("not yet implemented")
    def make_training_env(*args) -> compiler_gym.envs.CompilerEnv:
        """Make a reinforcement learning environment that cycles over the
        set of training benchmarks in use.
        Training benchmarks are received from server.
        """
        del args  # Unused env_config argument passed by ray
        return CycleOverBenchmarks(make_env(), train_benchmarks)

    tune.register_env("compiler_gym", make_training_env)

    # make this more easily runnable 
    analysis = tune.run(
        PPOTrainer,
        checkpoint_at_end=True,
        stop={
            "episodes_total": 1000,
        },
        config={
            "seed": 0xCC,
            "num_workers": 1,
            # Specify the environment to use, where "compiler_gym" is the name we
            # passed to tune.register_env().
            "env": "compiler_gym",
            # Reduce the size of the batch/trajectory lengths to match our short
            # training run.
            "rollout_fragment_length": 5,
            "train_batch_size": 5,
            "sgd_minibatch_size": 5,
            "framework": 'torch'
        }
    )
    return analysis
