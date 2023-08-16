# DQN Agent on 10x10 Board
# Stable Baselines Implementations procured from https://github.com/hill-a/stable-baselines

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import torch.nn as nn
from tensorboardX import SummaryWriter
from stable_baselines3.dqn.policies import DQNPolicy
import torch
import gymnasium as gym
from stable_baselines3 import DQN
import gym_examples
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.logger import Logger
import os
import random

# Function to Write Parameteric Values to Text File
def write_to_file(word):
    with open("unmasked_10_10_dqn_multiple_1.txt", "a") as file:
        file.write(word +"\n")

# Check if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    text = "cuda"
    write_to_file(text)
else:
    device = torch.device("cpu")
    text = "cpu"
    write_to_file(text)

# Defining a Custom Callback to Log Metrics to TensorBoard
class TensorBoardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorBoardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.n_episodes = 0

        # Episodic Data
        self.episode_cleared_lines = 0
        self.episode_cleared_lines_list = []
        self.episode_count = 0
        self.episode_total_reward = 0
        self.episode_total_reward_list = []
        self.episode_total_len = 0
        self.episode_total_len_list = []
        self.episodes_actions = []

        # Timestep Data
        self.timestep_cleared_lines = 0
        self.timestep_cleared_lines_list = []

        # Miscellanoues Initialisations
        self.writer = SummaryWriter()
        self.log_interval = 1000
        self.log_interval_distribution = 10000

    def _init_callback(self) -> None:
        # Episodic Data
        self.episode_cleared_lines = 0
        self.episode_cleared_lines_list = []
        self.episode_count = 0
        self.episodes_actions = []        
        self.episode_total_reward = 0
        self.episode_total_reward_list = []
        self.episode_total_len = 0
        self.episode_total_len_list = []

        # Timestep Data
        self.timestep_cleared_lines = 0
        self.timestep_cleared_lines_list = []

        # Miscellanoues Initialisation
        self.writer = SummaryWriter()
    
    # Function Implemnted at Each Timestep
    def _on_step(self) -> bool:
        self.episode_total_len += 1
        self.episode_total_reward += self.locals["rewards"][0]
        self.timestep_cleared_lines = self.locals["infos"][0]['cleared_lines']
        self.episode_cleared_lines += self.timestep_cleared_lines
        self.episodes_actions.append(self.locals["actions"][0])

        if self.locals.get('dones'):
            self.episode_count += 1
            self.episode_total_reward_list.append(self.episode_total_reward)
            self.episode_total_len_list.append(self.episode_total_len)
            self.episode_cleared_lines_list.append(self.episode_cleared_lines)

            t = self.num_timesteps
            if self.num_timesteps % self.log_interval == 0:
                # Logging Reward
                self.writer.add_scalar('timesteps/episode_reward', self.episode_total_reward, t)
                # Logging Lines cleared
                self.writer.add_scalar('timesteps/episode_lines_cleared', self.episode_cleared_lines, t)
                # Logging Episode length
                self.writer.add_scalar('timesteps/episode_length', self.episode_total_len, t)

                # Logging Average episodic lines cleared
                avg_lines_cleared = np.mean(self.episode_cleared_lines_list)
                self.writer.add_scalar("timesteps-average/average_episode_lines_cleared", avg_lines_cleared, t)
                # Logging Average episodic reward
                avg_ep_reward = np.mean(self.episode_total_reward_list)
                self.writer.add_scalar("timesteps-average/average_episode_reward", avg_ep_reward, t)
                # Logging Average episodic length
                avg_ep_len = np.mean(self.episode_total_len_list)
                self.writer.add_scalar("timesteps-average/average_episode_length", avg_ep_len, t)

                # Logging Training Stability - Variance            
                training_stability = np.var(self.episode_total_reward_list[-100:])
                self.writer.add_scalar("timesteps/training_stability", training_stability, t)

            # Logging Reward Distribution Data 
            if self.num_timesteps > 0 and len(self.episode_total_reward_list) % self.log_interval_distribution == 0:
                rewards_hist = (self.episode_total_reward_list)
                self.writer.add_histogram("reward_distribution", rewards_hist, t)

            # Logging Action Distribution Data 
            if self.num_timesteps > 0 and len(self.episode_total_reward_list) % self.log_interval_distribution == 0:
                episode_actions = (self.episodes_actions)
                self.writer.add_histogram("action_distribution", episode_actions, t)
           
            # Saving model
            if self.num_timesteps % self.log_interval == 0:
                for file_path in ["dqn_multiple_1.zip"]:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"Deleted existing file: {file_path}")
                model.save("dqn_multiple_1.zip")

            # Re-initialising for New Episode
            self.episode_cleared_lines = 0
            self.episode_total_len = 0
            self.episode_total_reward = 0

        return True
    


callback = TensorBoardCallback()

if __name__ == "__main__":
    timesteps = 10000000

    # Optuna Optimal Hyperparameters - 10 x 10 - DQN [Has to be changed accordingly for 10x20 Board]
    trial_number = 1
    reward_t = 7
    buffer_size = 815981
    learning_rate = 0.0001
    batch_size = 64
    gamma = 0.95
    tau = 0.05
    seed = random.randint(1, 1000)

    net_arch = [256, 256]
    activation_fn_name = "ReLU"

    exploration_fraction = 0.4771606694707132
    exploration_initial_eps = 0.1563199107511768
    exploration_final_eps = 0.22082328799820772

    if activation_fn_name == "ReLU":
        activation_fn = nn.ReLU
    elif activation_fn_name == "LeakyReLU":
        activation_fn = nn.LeakyReLU
    elif activation_fn_name == "Sigmoid":
        activation_fn = nn.Sigmoid
    elif activation_fn_name == "Softmax":
        activation_fn = nn.Softmax

    class CustomDQNMlpPolicy(DQNPolicy):
        def __init__(self, *args, net_arch=net_arch, activation_fn=activation_fn, **kwargs):
            super(CustomDQNMlpPolicy, self).__init__(*args, net_arch=net_arch, activation_fn=activation_fn, **kwargs)

    # Create and wrap the gym environment
    log_dir = 'unmasked_10_10_dqn_multiple_1/'
    env = gym.make("gym_examples/Tetris-Binary-v0", width = 10, height = 10, reward_type = reward_t) # Can be changed to 10x20 Board Configuration
    env = Monitor(env, log_dir)

    text = "Trial: " + str(trial_number) + " Reward: " + str(reward_t) + " Buffer Size: " + str(buffer_size) + " Learning rate: " + str(learning_rate) + " Batch Size: " + str(batch_size) + " Gamma: " +str(gamma) + " Tau: " + str(tau) + " Seed: " + str(seed) + " Net Arch: " + str(net_arch) + " Activation function: " +str(activation_fn) + " Exploration fraction: " + str(exploration_fraction) + " Exploration initial eps: "+str(exploration_initial_eps) + " Exploration final eps: "+ str(exploration_final_eps) 
    write_to_file(text)

    # Create the DQN agent
    model = DQN(
        CustomDQNMlpPolicy,
        env,
        buffer_size=buffer_size,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        seed=seed,
        verbose=1,
        tensorboard_log='unmasked_10_10_dqn_multiple_1/',
        device=device,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps
    )

    # Train the agent
    model.learn(total_timesteps=timesteps, callback=callback)

    # Evaluate the agent
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)
    text = "Score: " + str(mean_reward) + " for trail number " + str(trial_number)
    write_to_file(text)