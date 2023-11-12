import itertools
import os
import random
import shutil

import gymnasium as gym
import numpy as np
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

def train_agent(index: int, parameters: dict) -> dict:
    model_file = f"trained-agents/agent-{index}-{parameters['type']}"
    agent = make_agent(parameters)
    return agent.train(env, verbose=True, model_filename=model_file, training_filename=model_file+".måskejson")

if __name__ == "__main__":
    log.configure("lunar-training.log")
    with log.log_errors:
        shutil.rmtree("trained-agents", ignore_errors=True)
        os.makedirs("trained-agents", exist_ok=True)
        env = gym.make('LunarLander-v2')
        parameter_sets = make_parameter_sets()
        random.shuffle(parameter_sets)
        agents_per_parameter = 10
        log(
            f"Got {len(parameter_sets):,} parameter sets",
            f"Training {agents_per_parameter * len(parameter_sets):,} agents",
        )
        args = list(enumerate(parameter_sets * agents_per_parameter))
        for i, parameters in enumerate(parameter_sets * agents_per_parameter):
            log(f"Parameter set {i + 1:,} / {len(parameter_sets)}")
            results = train_agent(i, parameters)
            log("Mean of last 100 episodes: %.2f" % np.mean(results["epsiode_returns"][-100:]))
