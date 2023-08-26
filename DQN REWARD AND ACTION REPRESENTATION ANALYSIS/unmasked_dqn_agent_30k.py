# Stable Baselines Implementations procured from https://github.com/hill-a/stable-baselines
import optuna
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from gymnasium import spaces
from stable_baselines3.dqn.policies import DQNPolicy
import torch
import gymnasium as gym
from stable_baselines3 import DQN
import gym_examples

# Function to Write Parameteric Values to Text File for Each Trial
def write_to_file(word):
    with open("unmasked_binary_30k_10_10.txt", "a") as file:
        file.write(word +"\n")

# Check if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    text = "cuda"
    write_to_file(text)
else:
    device = torch.device("cpu")

# Creating a Custom Callback Function to Stop Training on Reaching Maximum Episodes -- code adapted from: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html -> StopTrainingOnMaxEpisodes Callback
class StopTrainingOnMaxEpisodes(BaseCallback):
    """
    Stop the training once a maximum number of episodes are played.

    For multiple environments presumes that, the desired behavior is that the agent trains on each env for ``max_episodes``
    and in total for ``max_episodes * n_envs`` episodes.

    :param max_episodes: Maximum number of episodes to stop training.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about when training ended by
        reaching ``max_episodes``
    """

    def __init__(self, max_episodes: int, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.max_episodes = max_episodes
        self._total_max_episodes = max_episodes
        self.n_episodes = 0

        # Episodic Data
        self.episode_cleared_lines = 0
        self.episode_cleared_lines_list = []
        self.episode_count = 0

        self.episode_total_reward = 0
        self.episode_total_reward_list = []
        self.episode_total_len = 0
        self.episode_total_len_list = []

        # Timestep Data
        self.timestep_cleared_lines = 0
        self.timestep_cleared_lines_list = []

        # Miscellanoues Initialisations
        self.writer = SummaryWriter()

    def _init_callback(self) -> None:
        # At start set total max according to number of envirnments
        self._total_max_episodes = self.max_episodes * self.training_env.num_envs

        # Episodic Data
        self.episode_cleared_lines = 0
        self.episode_cleared_lines_list = []
        self.episode_count = 0
        
        self.episode_total_reward = 0
        self.episode_total_reward_list = []
        self.episode_total_len = 0
        self.episode_total_len_list = []

        # Timestep Data
        self.timestep_cleared_lines = 0
        self.timestep_cleared_lines_list = []

        # Miscellanoues Initialisations
        self.writer = SummaryWriter()

    # Function Implemnted at Each Timestep
    def _on_step(self) -> bool:
        self.episode_total_len += 1
        self.episode_total_reward += self.locals["rewards"][0]
        self.timestep_cleared_lines = self.locals["infos"][0]['cleared_lines']
        self.timestep_cleared_lines_list.append(self.timestep_cleared_lines)
        self.episode_cleared_lines += self.timestep_cleared_lines
        
        # Check if the episode is over
        if self.locals.get('dones'):
            self.episode_count += 1
            self.episode_total_reward_list.append(self.episode_total_reward)
            self.episode_total_len_list.append(self.episode_total_len)            
            self.episode_cleared_lines_list.append(self.episode_cleared_lines)
            self.episode_cleared_lines = 0
            self.timestep_cleared_lines = 0
            self.episode_total_len = 0
            self.episode_total_reward = 0

        # Check that the `dones` local variable is defined
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
        self.n_episodes += np.sum(self.locals["dones"]).item()
        continue_training = self.episode_count < self._total_max_episodes # Checks if Max Episode is Reached

        if self.verbose >= 1 and not continue_training:
            mean_episodes_per_env = self.episode_count / self.training_env.num_envs
            mean_ep_str = (
                f"with an average of {mean_episodes_per_env:.2f} episodes per env" if self.training_env.num_envs > 1 else ""
            )

            print(
                f"Stopping training with a total of {self.num_timesteps} steps because the "
                f"{self.locals.get('tb_log_name')} model reached max_episodes={self.max_episodes}, "
                f"by playing for {self.episode_count} episodes "
                f"{mean_ep_str}"
            )
        return continue_training

    # Function Implemnted at The End of Training    
    def _on_training_end(self) -> None:
       
       # Logging
        self.writer.add_figure('episode/lines_cleared_plot_1', self.plot_lines_cleared_episode(self.episode_cleared_lines_list), self.episode_count)
        self.writer.add_figure('timestep/lines_cleared_plot_1', self.plot_lines_cleared_timestep(self.timestep_cleared_lines_list), self.num_timesteps)
        

        for t, lines in enumerate(self.episode_cleared_lines_list):
            self.writer.add_scalar('episode/lines_cleared_plot', lines, t)
        
        for t, lines in enumerate(self.timestep_cleared_lines_list):
            self.writer.add_scalar('timestep/lines_cleared_plot', lines, t)
        
        for t, length in enumerate(self.episode_total_len_list):
            self.writer.add_scalar('episode/total_len', length, t)
        
        for t, reward in enumerate(self.episode_total_reward_list):
            self.writer.add_scalar('episode/total_reward', reward, t)
        
        avg_lines_cleared = np.mean(self.episode_cleared_lines_list)
        avg_timestep_lines_cleared = np.mean(self.timestep_cleared_lines_list)
        ep = "Average lines cleared across 30000 episodes:"
        self.writer.add_scalar(ep, avg_lines_cleared)
        self.writer.add_scalar("Average lines cleared timesteps:", avg_timestep_lines_cleared)
        self.writer.close()
    

    def plot_lines_cleared_episode(self, lines_cleared):
        episode_number = range(self.episode_count)
        
        plt.plot(episode_number, lines_cleared)
        plt.xlabel('Episode Number')
        plt.ylabel('Lines Cleared')
        plt.title('Lines Cleared vs Episode Number')
        plt.grid(True)
        return plt.gcf()
    
    def plot_lines_cleared_timestep(self, lines_cleared):
        episode_number = range(self.num_timesteps)
        
        plt.plot(episode_number, lines_cleared)
        plt.xlabel('Timestep')
        plt.ylabel('Lines Cleared')
        plt.title('Lines Cleared vs Timestep')
        plt.grid(True)
        return plt.gcf()


# Creating the Callback
callback = StopTrainingOnMaxEpisodes(30000)

# Running Each Agent with One Reward Thrice --Unmasked Actions
rew = list(range(1, 12))
for i in rew:
    for j in range(1, 4):

        # Create your Tetris environment
        env = gym.make("gym_examples/Tetris-Binary-v0", width = 10, height = 10, reward_type = i)

        # Create the DQN agent
        model = DQN('MlpPolicy', env, verbose=1, tensorboard_log='unmasked_binary_30k_10_10/')

        # Train the agent
        model.learn(total_timesteps=1000000000, callback=callback) 

        model.save("unmasked_binary_30k_10_10_"+str(i)+"_"+str(j)+".zip")

        word_loop = "Done: " + str(i) + " " + str(j)
        write_to_file(word_loop)