# Import Statements
from AgentPreparation import AgentPreparation
import torch
import os
import numpy as np
import random
import gymnasium as gym
import gym_examples
import json
import time

from utilities.Utility_Functions import create_actor_distribution 

# Function to Write Trail Related Details
def write_to_file(data, filename):
    with open(filename, "a") as file:
        json.dump(data, file)
        file.write('\n')
        file.write('\n')

def evaluation(config):
    print("Evaluation")
    filename = "evaluation.txt"

    config["seed"] = 1 #random.randint(1, 1000)

    '''Setting Seeds and Torch Parameters for the Required Enviornments for Reproducability'''

    # Setting Seed for Random and Numpy 
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    # Setting Seed for OS
    os.environ["PYTHONHASHSEED"] = str(config["seed"])

    # Setting Seed for Torch
    torch.backends.cudnn.deterministic = True # Enforces Deterministic Behavior in CuDNN for Reproducability
    torch.backends.cudnn.benchmark = False # Disables CuDNN's Benchmarking Functionality to Choose the Best Algorithm for Reproducability
    torch.manual_seed(config["seed"])

    # Checking if GPU is Available and Setting Seed for Reproducability
    if torch.cuda.is_available():
        config["use_GPU"] = True
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.manual_seed(config["seed"])
    else:
        config["use_GPU"] = False

    agent = AgentPreparation(config)

    # Load the state dictionary from the .pt file
    agent.actor_local.load_state_dict(torch.load("11_3_actor_local_network.pt"))
    agent.actor_local.eval()  # Set the model to evaluation mode


    print("Actor", agent.actor_local)

    # Create your Tetris environment
    env = gym.make("gym_examples/Tetris-Binary-v0", width = 10, height = 10, reward_type = 11)

    # Define a function to record the environment
    def record_environment(env, num_episodes=30):
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            cleared_lines = 0
            length = 0
            while not done:
                # Picking Action
                # Agent in Evaluation Mode and Training Mode Make Actor choose the Action
                state = obs
                state = torch.FloatTensor([state]).to(agent.device)
                if len(state.shape) == 1: state = state.unsqueeze(0)
                action_probabilities = agent.actor_local(state)
                max_probability_action = torch.argmax(action_probabilities, dim=-1)
                action_distribution = create_actor_distribution(agent.action_types, action_probabilities, agent.action_size)
                action = action_distribution.sample().cpu()
                # Dealing with log 0
                z = action_probabilities == 0.0
                z = z.float() * 1e-8
                log_action_probabilities = torch.log(action_probabilities + z)

                if evaluation == False: 
                    action = action
                else:
                    with torch.no_grad():
                        z, action = (action_probabilities, log_action_probabilities), max_probability_action
                action = action.detach().cpu().numpy()
                action = action[0]
                obs, reward, done, _, info = env.step(action)
                tetrimino = str(obs[-1])
                text = "action: " + str(action) + " tetrimino: " + tetrimino
                write_to_file(text, filename)
                # print("action", action)
                # env.render()
                # time.sleep(2)
                episode_reward += reward
                cleared_lines += info["cleared_lines"]
                length += 1
            print("Episode", episode, episode_reward, cleared_lines, length)
            text = "Episode End: " + str(episode) + " Reward: " + str(episode_reward) + " Cleared-lines: " +str(cleared_lines) + " Length: " +str(length)
            write_to_file(text, filename)
        env.close()

    # Record the environment using the loaded model
    record_environment(env)

    # Evaluate the performance of the model
    def evaluate_model(env, num_episodes=100):
        rewards = []
        lines = []
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            cleared_lines = 0
            while not done:
                # Picking Action
                # Agent in Evaluation Mode and Training Mode Make Actor choose the Action
                state = obs
                state = torch.FloatTensor([state]).to(agent.device)
                if len(state.shape) == 1: state = state.unsqueeze(0)
                action_probabilities = agent.actor_local(state)
                max_probability_action = torch.argmax(action_probabilities, dim=-1)
                action_distribution = create_actor_distribution(agent.action_types, action_probabilities, agent.action_size)
                action = action_distribution.sample().cpu()
                # Dealing with log 0
                z = action_probabilities == 0.0
                z = z.float() * 1e-8
                log_action_probabilities = torch.log(action_probabilities + z)

                if evaluation == False: 
                    action = action
                else:
                    with torch.no_grad():
                        z, action = (action_probabilities, log_action_probabilities), max_probability_action
                action = action.detach().cpu().numpy()
                action = action[0]
                obs, reward, done, _, info = env.step(action)
                episode_reward += reward
                cleared_lines += info["cleared_lines"]
            rewards.append(episode_reward)
            lines.append(cleared_lines)
        average_reward = sum(rewards) / num_episodes
        average_lines = sum(lines) / num_episodes
        str1 = ("Average reward over {} episodes: {:.2f}".format(num_episodes, average_reward))
        str2 = ("Average lines over {} episodes: {:.2f}".format(num_episodes, average_lines))
        write_to_file(str1, filename)
        write_to_file(str2, filename)

    # Evaluate the loaded model
    evaluate_model(env)
