#!/usr/bin/env python

import gymnasium as gym
import os
import argparse
import torch
import numpy as np
import itertools 
import h5py 

import lunar_lander.agent_class as agent

def load_agent(input_filename: str, dqn: bool):
    with open(input_filename,'rb') as f:
        input_dictionary = torch.load(f)

    dict_keys = np.array(list(input_dictionary.keys())).astype(int)
    max_index = np.max(dict_keys)
    input_dictionary = input_dictionary[max_index]

    parameters = input_dictionary['parameters']
    # Instantiate agent class
    if dqn:
        my_agent = agent.dqn(parameters=parameters)
    else:
        my_agent = agent.actor_critic(parameters=parameters)

    my_agent.load_state(state=input_dictionary)
    return my_agent

def run_and_save_simulations(env, # environment
                            input_filename,output_filename,N=1000,
                            dqn=False):
    #
    # load trained model
    input_dictionary = torch.load(open(input_filename,'rb'))
    dict_keys = np.array(list(input_dictionary.keys())).astype(int)
    max_index = np.max(dict_keys)
    input_dictionary = input_dictionary[max_index] # During training we 
    # periodically store the state of the neural networks. We now use
    # the latest state (i.e. the one with the largest episode number), as 
    # for any succesful training this is the state that passed the stopping
    # criterion
    #
    # instantiate agent
    parameters = input_dictionary['parameters']
    # Instantiate agent class
    if dqn:
        my_agent = agent.dqn(parameters=parameters)
    else:
        my_agent = agent.actor_critic(parameters=parameters)
    my_agent.load_state(state=input_dictionary)
    #
    # instantiate environment
    env = gym.make('LunarLander-v2')
    #
    durations = []
    returns = []
    status_string = ("Run {0} of {1} completed with return {2:<5.1f}. Mean "
            "return over all episodes so far = {3:<6.1f}            ")
    # run simulations
    for i in range(N):
        # reset environment, duration, and reward
        state, info = env.reset()
        episode_return = 0.
        #
        for n in itertools.count():
            #
            action = my_agent.act(state)
            #
            state, step_reward, terminated, truncated, info = env.step(action)
            #
            done = terminated or truncated
            episode_return += step_reward
            #
            if done:
                #
                durations.append(n+1)
                returns.append(episode_return)
                #
                if verbose:
                    if i < N-1:
                        end ='\r'
                    else:
                        end = '\n'
                    print(status_string.format(i+1,N,episode_return,
                                        np.mean(np.array(returns))),
                                    end=end)
                break
    #
    dictionary = {'returns':np.array(returns),
                'durations':np.array(durations),
                'input_file':input_filename,
                'N':N}
        
    with h5py.File(output_filename, 'w') as hf:
        for key, value in dictionary.items():
            hf.create_dataset(str(key), 
                data=value)
    

# # Create environment
# env = gym.make('LunarLander-v2')

# run_and_save_simulations(env=env,
#                             input_filename=input_filename,
#                             output_filename=output_filename,
#                             N=N,
#                             dqn=dqn)
