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
    with open("masked_binary_30k_10_10.txt", "a") as file:
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
        ep = "Average lines cleared across 50000 episodes:"
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

# Inheriting the DQN Class to Incorporate Masking Actions
class MaskedDQN(DQN):
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                _, masks, _ = self.env.envs[0].get_next_states()
                action = np.array([self.action_space.sample(masks) for _ in range(n_batch)])
                #print("PREDICT", action, masks)
            else:
                _, masks, _ = self.env.envs[0].get_next_states()
                action = np.array(self.action_space.sample(masks))
                #print("PREDICT-1", action, masks)
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state
    
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase 
            _, masks, _ = self.env.envs[0].get_next_states()
            unscaled_action = np.array([self.action_space.sample(masks) for _ in range(n_envs)])
            #print("unscaled action", unscaled_action, masks)
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

# Running Each Agent with One Reward Thrice --Masked Actions
rew = list(range(1, 12))
for i in rew:
    for j in range(1, 4):

        # Create your Tetris environment
        env = gym.make("gym_examples/Tetris-Binary-v0", width = 10, height = 10, reward_type = i)

        # Create the Masked DQN agent
        model = MaskedDQN('MlpPolicy', env, verbose=1, tensorboard_log='masked_binary_30k_10_10/')

        # Train the agent
        model.learn(total_timesteps=1000000000, callback=callback) 

        model.save("masked_binary_30k_10_10_"+str(i)+"_"+str(j)+".zip")

        word_loop = "Done: " + str(i) + " " + str(j)
        write_to_file(word_loop)