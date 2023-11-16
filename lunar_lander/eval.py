import os
import pickle
import sys
from glob import glob as glob  # glob

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots
from tqdm import tqdm

sys.path.append("..")

from multitrainer import path
from run_agent import load_agent
from agent_class import agent_base


n = 1000
env = gym.make('LunarLander-v2')

def evaluate(agent_path: str):
    print(agent_path)
    try:
        agent = load_agent(agent_path)
    except (EOFError, RuntimeError) as e:
        print("Failed for %s" % agent_path, e)
        return
    rewards = list()
    ticks = list()
    for i in range(n):
        obs, _ = env.reset()
        steps = 0
        total_reward = 0
        is_terminal = False
        while not is_terminal and steps < 1000:
            action = agent.act(obs)
            obs, reward, is_terminal, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        rewards.append(total_reward)
        ticks.append(steps)

    return rewards, ticks

if __name__ == "__main__":
    paths = glob(os.path.join(path, "*.best.pt"))

    all_paths = list()
    all_rewards = list()
    all_gameticks = list()

    print("Starting")
    print(f"{len(paths):,} agents")
    import torch
    import multiprocessing as mp
    torch.set_num_threads(1)

    with mp.Pool() as pool:
        results = pool.map(evaluate, paths, chunksize=1)

    for i, rg in enumerate(results):
        if rg is None:
            continue
        r, g = rg
        all_rewards.append(r)
        all_gameticks.append(g)
        all_paths.append(paths[i])

    # with open("eval.pkl", "rb") as f:
    #     all_paths, all_rewards, all_gameticks = pickle.load(f)

    all_rewards = np.array(all_rewards)
    all_gameticks = np.array(all_gameticks)
    with plots.Figure("eval.png", figsize=(20, 10)):
        plt.subplot(121)
        mean_rewards = all_rewards.mean(axis=1)
        sorted_rewards = np.sort(all_rewards, axis=1)
        mean_gameticks = all_gameticks.mean(axis=1)
        best = mean_rewards.argsort()[-15:]
        for b in best:
            print(mean_rewards[b], sorted_rewards[b][n//4], all_rewards.min(axis=1)[b], all_paths[b])
        plt.scatter(mean_gameticks, mean_rewards)
        plt.grid()
        plt.xlabel("Game ticks")
        plt.ylabel("Reward")
        plt.title("Mean")

        plt.subplot(122)
        min_rewards = all_rewards.min(axis=1)
        min_gameticks = all_gameticks.max(axis=1)
        plt.scatter(min_gameticks, min_rewards)
        plt.grid()
        plt.xlabel("Game ticks")
        plt.ylabel("Reward")
        plt.title("Max game ticks / min reward")
        plt.ylim(bottom=-50)

    with open("eval.pkl", "wb") as f:
        pickle.dump((all_paths, all_rewards, all_gameticks), f)
