# Stable Baselines Implementations procured from https://github.com/hill-a/stable-baselines
from stable_baselines3 import DQN
import gymnasium as gym
from stable_baselines3 import DQN
import gym_examples
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from gymnasium import spaces
import numpy as np
import time
from tensorboardX import SummaryWriter

# Function to Write Evaluation Results
def write_to_file(word):
    with open("unmasked_binary_evaluation_20.txt", "a") as file:
        file.write(word +"\n")

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

# Iterating over all 11 Rewards
for i in range(1, 12):
    for j in range(1, 2):
        text = "Reward: " + str(i)
        write_to_file(text)

        # Settting Model Files
        model_filename = "unmasked_20/unmasked_binary_30k_20_" + str(i) + "_" +str(j) +".zip"
        logdir = "unmasked_dqn_"+ str(i) + "_" +str(j)
        writer = SummaryWriter(log_dir = logdir)

        # Load the DQN model from the saved file
        model = DQN.load(model_filename) # Use MaskedDQN for the Masked Version

        # Create your Tetris environment
        env = gym.make("gym_examples/Tetris-Binary-v0", width = 10, height = 10, reward_type = i) # Can be changed to 10x20 Version

        # Define a function to record the environment
        def record_environment(env, model, num_episodes=500):
            for episode in range(num_episodes):
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                cleared_lines = 0
                actions = []
                ep_len = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = env.step(action)
                    actions.append(str(action))
                    #env.render()
                    #time.sleep(2)
                    episode_reward += reward
                    cleared_lines += info["cleared_lines"]
                    ep_len += 1
                #print("Episode", episode, episode_reward, cleared_lines)
                list_as_string = ",".join(str(item) for item in actions)
                text = "Episode: " + str(episode) + " Reward: " +str(episode_reward) +" Cleared: " +str(cleared_lines) + " Actions: " + list_as_string + " Length: " +str(ep_len)
                writer.add_scalar("episode/total_reward", episode_reward, episode)
                writer.add_scalar("episode/total_len", ep_len, episode)
                writer.add_scalar("episode/lines_cleared_plot", (cleared_lines), episode)
                writer.add_scalar("episode/average_lines_cleared_plot", np.mean(cleared_lines), episode)
                write_to_file(text)
            env.close()

        # Record the environment using the loaded model
        record_environment(env, model)

        # Evaluate the performance of the model
        def evaluate_model(env, model, num_episodes=100):
            rewards = []
            lines = []
            for episode in range(num_episodes):
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                cleared_lines = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = env.step(action)
                    episode_reward += reward
                    cleared_lines += info["cleared_lines"]
                rewards.append(episode_reward)
                lines.append(cleared_lines)
            average_reward = sum(rewards) / num_episodes
            average_lines = sum(lines) / num_episodes
            text = "Average reward over 100 episodes " + str(average_reward)
            write_to_file(text)
            text = "Average lines over 100 episodes " + str(average_lines)
            write_to_file(text)
            print("Average reward over {} episodes: {:.2f}".format(num_episodes, average_reward))
            print("Average lines over {} episodes: {:.2f}".format(num_episodes, average_lines))

        # Evaluate the loaded model
        evaluate_model(env, model)
