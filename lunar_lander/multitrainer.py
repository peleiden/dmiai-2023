import os
import shutil

import gymnasium as gym
from pelutils import log

from agent_class import make_agent


def make_parameter_sets() -> list[dict]:
    return [
        {'type': 'dqn', 'N_state': 8, 'N_actions': 4},
        {'type': 'ddqn', 'N_state': 8, 'N_actions': 4},
        {'type': 'ac', 'N_state': 8, 'N_actions': 4},
    ]


def train_agent(index: int, parameters: dict):
    model_file = f"trained-agents/agent-{index}-{parameters['type']}"
    agent = make_agent(parameters)
    agent.train(env, verbose=True, model_filename=model_file, training_filename=model_file+".måskejson")

if __name__ == "__main__":
    log.configure("lunar-training.log")
    with log.log_errors:
        shutil.rmtree("trained-agents", ignore_errors=True)
        os.makedirs("trained-agents", exist_ok=True)
        env = gym.make('LunarLander-v2')
        parameter_sets = make_parameter_sets()
        agents_per_parameter = 1
        log(
            f"Got {len(parameter_sets):,} parameter sets",
            f"Training {agents_per_parameter * len(parameter_sets):,} agents",
        )
        for i, parameters in enumerate(parameter_sets):
            log(f"Parameter set {i + 1:,} / {len(parameter_sets)}")
            for j in range(agents_per_parameter):
                index = i * agents_per_parameter + j
                train_agent(index, parameters)
