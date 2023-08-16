import gymnasium as gym
import gym_examples
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

# Function to Plot Lines Cleared vs Episode Number
def plot_lines_cleared_episode(episode_lines_cleared):
    episode_number = range(len(episode_lines_cleared))
    
    plt.plot(episode_number, episode_lines_cleared)
    plt.xlabel('Episode Number')
    plt.ylabel('Lines Cleared')
    plt.title('Lines Cleared vs Episode Number')
    plt.grid(True)
    return plt.gcf()

# Function to Plot Lines Cleared vs Timestep  
def plot_episode_rewards(episode_rewards):
    episode_number = range(len(episode_rewards))
    
    plt.plot(episode_number, episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title('Rewards vs Episode')
    plt.grid(True)
    return plt.gcf()

# Iterating over each Reward Function
for i in range(1, 12):
    # Create your Tetris environment
    env = gym.make("gym_examples/Tetris-Binary-v0", width = 10, height = 10, reward_type = i)

    num_episodes = 30000
    logdir = "./random_agent_" + str(i)

    writer = SummaryWriter(logdir = logdir)

    episode_rewards = []
    episode_cleared_lines = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        cleared_lines = 0
        while not done:
            action = env.action_space.sample() # Sampling from Unmasked Actions
            obs, reward, done, _, info = env.step(action)
            #env.render()
            episode_reward += reward
            cleared_lines += info["cleared_lines"]
        print("Episode", episode, episode_reward, cleared_lines)
        episode_cleared_lines.append(cleared_lines)
        episode_rewards.append(episode_reward)
    env.close()

    # Logging
    for t, reward in enumerate(episode_rewards):
        writer.add_scalar('episode/rewards', reward, t)
    
    for t, lines in enumerate(episode_cleared_lines):
        writer.add_scalar('episode/lines_cleared', lines, t)

    avg_lines_cleared = np.mean(episode_cleared_lines)
    avg_reward = np.mean(episode_rewards)
    ep = "Average lines cleared across " + str(len(episode_cleared_lines)) + " episodes:"
    writer.add_scalar(ep, avg_lines_cleared)
    writer.add_scalar("Average reward 30000 episodes:", avg_reward)
    print("Done", i)

    writer.close()
