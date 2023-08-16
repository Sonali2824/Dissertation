import gymnasium as gym
import gym_examples
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_examples

# Function to Plot Lines Cleared vs Episode Number
def plot_lines_cleared_episode(self, lines_cleared):
    episode_number = range(self.episode_count)
    
    plt.plot(episode_number, lines_cleared)
    plt.xlabel('Episode Number')
    plt.ylabel('Lines Cleared')
    plt.title('Lines Cleared vs Episode Number')
    plt.grid(True)
    return plt.gcf()

# Function to Plot Lines Cleared vs Timestep  
def plot_lines_cleared_timestep(self, lines_cleared):
    episode_number = range(self.num_timesteps)
    
    plt.plot(episode_number, lines_cleared)
    plt.xlabel('Timestep')
    plt.ylabel('Lines Cleared')
    plt.title('Lines Cleared vs Timestep')
    plt.grid(True)
    return plt.gcf()

# Iterating over each Reward Function
for i in range(1, 12):
    # Create your Tetris environment
    env = gym.make("gym_examples/Tetris-Binary-v0", width = 10, height = 10, reward_type = i)

    num_episodes = 30000
    logdir = "./random_agent_1_" + str(i)

    writer = SummaryWriter(logdir = logdir)

    episode_rewards = []
    episode_cleared_lines = []
    episode_lens = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        cleared_lines = 0
        ep_len = 0
        while not done:
            _, masks, _ = env.get_next_states()
            action = env.action_space.sample(masks) # Sampling from Masked Actions
            obs, reward, done, _, info = env.step(action)
            #env.render()
            episode_reward += reward
            cleared_lines += info["cleared_lines"]
            ep_len += 1
        episode_cleared_lines.append(cleared_lines)
        episode_rewards.append(episode_reward)
        episode_lens.append(ep_len)
    env.close()

    # Logging
    for t, reward in enumerate(episode_rewards):
        writer.add_scalar('episode/total_reward', reward, t)
    
    for t, lines in enumerate(episode_cleared_lines):
        writer.add_scalar('episode/lines_cleared_plot', lines, t)
    
    for t, len in enumerate(episode_lens):
        writer.add_scalar('episode/total_len', len, t)

    avg_lines_cleared = np.mean(episode_cleared_lines)
    avg_reward = np.mean(episode_rewards)
    ep = "Average lines cleared across 30000 episodes:"
    writer.add_scalar(ep, avg_lines_cleared)
    writer.add_scalar("Average reward 30000 episodes:", avg_reward)
    print("Done", i)

    writer.close()
