import os
import pickle
import sys
from glob import glob as glob  # glob
import itertools

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots
from tqdm import tqdm

sys.path.append("..")

from run_agent import load_agent
from agent_class import agent_base
from ensemble import act_ensemble


n = 200
env = gym.make('LunarLander-v2')

paths = glob("gode-agenter/*")
agents = [load_agent(x) for x in paths]

def evaluate(agents: list[agent_base]):
    rewards = list()
    ticks = list()
    for i in tqdm(range(n), position=1):
        obs, _ = env.reset(seed=i)
        steps = 0
        total_reward = 0
        is_terminal = False
        while not is_terminal and steps < 1000:
            action = act_ensemble(obs, agents)
            obs, reward, is_terminal, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        rewards.append(total_reward)
        ticks.append(steps)

    return rewards, ticks

def evaluate_single_agents():
    rewards = np.empty((len(agents), n))
    ticks = np.empty((len(agents), n), dtype=int)
    for i, agent in tqdm(enumerate(agents)):
        r, g = evaluate([agent])
        rewards[i] = r
        ticks[i] = g
    return rewards, ticks

def evaluate_ensemble(sorted_agents: list[agent_base]):
    pass

if __name__ == "__main__":

    single_rewards, single_ticks = evaluate_single_agents()
    index = np.argsort(single_rewards.mean(axis=1))[::-1]
    paths = [paths[i] for i in index]
    agents = [agents[i] for i in index]
    single_rewards = single_rewards[index]
    single_ticks = single_ticks[index]

    with plots.Figure("single-agents.png"):
        for i in range(len(agents)):
            print(
                "%i" % i,
                paths[i],
                "%.2f" % single_rewards[i].mean(),
                "%.2f" % np.sort(single_rewards[i])[n//4],
                "%.2f" % single_rewards[i].min(),
            )
            plt.scatter(np.full(n, i), single_rewards[i])
        plt.grid()

    agent_combs = list()
    ensemble_rewards = list()
    ensemble_ticks = list()
    used_combs = set()
    for i, (j, k, l) in tqdm(enumerate(itertools.product(range(len(agents)), range(len(agents)), range(len(agents))))):
        if len({j, k, l}) < 3:
            continue
        if tuple(sorted((j, k, l))) in used_combs:
            continue
        used_combs.add(tuple(sorted((j, k, l))))
        rewards, ticks = evaluate([agents[j], agents[k], agents[l]])
        agent_combs.append((j, k, l))
        ensemble_rewards.append(rewards)
        ensemble_ticks.append(ticks)

    ensemble_rewards = np.array(ensemble_rewards)
    ensemble_ticks = np.array(ensemble_ticks)

    with plots.Figure("ensemble-agents.png"):
        for i in range(len(agent_combs)):
            print(
                agent_combs[i],
                "%.2f" % ensemble_rewards[i].mean(),
                "%.2f" % np.sort(ensemble_rewards[i])[n//4],
                "%.2f" % ensemble_rewards[i].min(),
            )
            plt.scatter(np.full(n, i), ensemble_rewards[i])
        plt.grid()
        plt.xticks(np.arange(len(ensemble_rewards)), [str(x) for x in agent_combs])
