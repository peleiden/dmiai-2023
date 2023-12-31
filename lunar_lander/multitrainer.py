import itertools
import multiprocessing as mp
import os
import random
import shutil
from pprint import pformat

import gymnasium as gym
import numpy as np
import torch
from pelutils import log

from agent_class import make_agent

work3 = False
path = "/work3/s183912/trained-agents" if work3 else "/home/asger/Skrivebord/best-agents/best-agents/"

def make_parameter_sets() -> list[dict]:

    types = ["dqn", "ddqn", "ac"]
    layers = [[256, 128, 64], [512, 256, 256], [512, 256, 256, 128]]
    bnorms = [False]
    memories = [20000]
    training_strides = [2, 3, 4]
    batch_sizes = [32, 64]
    discount_factors = [0.99, 0.995]
    epsilons = [0.1, 0.05, 0.03]
    taus = [1e-1, 1e-2, 1e-3]

    parameter_sets = list()

    for type, layer, epsilon, bnorm, memory, training_stride, batch_size, discount_factor, tau in itertools.product(types, layers, epsilons, bnorms, memories, training_strides, batch_sizes, discount_factors, taus):

        parameters = {
            'type': type,
            'N_state': 8,
            'N_actions': 4,
            'layers': [8, *layer, 4],
            'bnorm': bnorm,
            'epsilon_1': epsilon,
            'target_net_update_tau': tau,
            #
            'n_memory': memory,
            'training_stride': training_stride,
            'batch_size': batch_size,
            'saving_stride': 500,
            #
            'n_episodes_max': 10000,
            'n_solving_episodes': 69,
            'solving_threshold_min': 250,
            'solving_threshold_mean': 300,
            #
            'discount_factor': discount_factor,
        }
        parameter_sets.append(parameters)

    return parameter_sets

def train_agent(args: tuple) -> dict:
    index, parameters = args
    if not work3:
        log(pformat(parameters))
    log(f"Training agent {index:,}")
    model_file = f"{path}/agent-{index}-{parameters['type']}"
    agent = make_agent(parameters)
    results = agent.train(env, verbose=not work3, model_filename=model_file, training_filename=model_file+".måskejson")
    log(
        "Agent %i mean return of last 100 episodes: %.2f" % (index, np.mean(results["epsiode_returns"][-100:])),
        "Agent %i min. return of last 100 episodes: %.2f" % (index, np.min(results["epsiode_returns"][-100:])),
    )
    return results

if __name__ == "__main__":
    with log.log_errors:
        log.configure("lunar-training.log")
        torch.set_num_threads(1)
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
        env = gym.make('LunarLander-v2')
        parameter_sets = make_parameter_sets()
        random.shuffle(parameter_sets)
        agents_per_parameter = 10
        log(
            f"Got {len(parameter_sets):,} parameter sets",
            f"Training {agents_per_parameter * len(parameter_sets):,} agents",
        )
        args = list(enumerate(parameter_sets * agents_per_parameter))
        if work3:
            with mp.Pool(128) as pool:
                pool.map(train_agent, args, chunksize=1)
        else:
            for arg in args:
                train_agent(arg)
