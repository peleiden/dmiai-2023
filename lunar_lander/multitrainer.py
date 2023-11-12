import itertools
import multiprocessing as mp
import os
import random
import shutil

import gymnasium as gym
import numpy as np
import torch
from pelutils import log

from agent_class import make_agent


def make_parameter_sets() -> list[dict]:

    types = ["dqn", "ddqn", "ac"]
    layers = [[64], [128, 32], [128, 64, 32]]
    memories = [5000, 20000, 50000]
    training_strides = [3, 5, 10]
    batch_sizes = [16, 32, 128]
    discount_factors = [0.98, 0.99, 0.995]

    parameter_sets = list()

    for type, layer, memory, training_stride, batch_size, discount_factor in itertools.product(types, layers, memories, training_strides, batch_sizes, discount_factors):

        parameters = {
            'type': type,
            'N_state': 8,
            'N_actions': 4,
            'layers': [8, *layer, 4],
            #
            'n_memory': memory,
            'training_stride': training_stride,
            'batch_size': batch_size,
            'saving_stride': 100,
            #
            'n_episodes_max': 20000,
            'n_solving_episodes': 100,
            'solving_threshold_min': 250,
            'solving_threshold_mean': 270,
            #
            'discount_factor': discount_factor,
        }
        parameter_sets.append(parameters)

    return parameter_sets

def train_agent(args: tuple) -> dict:
    index, parameters = args
    log(f"Training agent {index:,}")
    model_file = f"/work3/s183912/trained-agents/agent-{index}-{parameters['type']}"
    agent = make_agent(parameters)
    results = agent.train(env, verbose=False, model_filename=model_file, training_filename=model_file+".måskejson")
    log("Agent %i mean return of last 100 episodes: %.2f" % (index, np.mean(results["epsiode_returns"][-100:])))
    return results

if __name__ == "__main__":
    with log.log_errors:
        log.configure("lunar-training.log")
        torch.set_num_threads(1)
        shutil.rmtree("/work3/s183912/trained-agents", ignore_errors=True)
        os.makedirs("/work3/s183912/trained-agents", exist_ok=True)
        env = gym.make('LunarLander-v2')
        parameter_sets = make_parameter_sets()
        random.shuffle(parameter_sets)
        agents_per_parameter = 10
        log(
            f"Got {len(parameter_sets):,} parameter sets",
            f"Training {agents_per_parameter * len(parameter_sets):,} agents",
        )
        args = list(enumerate(parameter_sets * agents_per_parameter))
        with mp.Pool(mp.cpu_count()) as pool:
            pool.map(train_agent, args, chunksize=1)
