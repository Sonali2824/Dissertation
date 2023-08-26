# Stable Baselines Implementations procured from https://github.com/hill-a/stable-baselines
from stable_baselines3 import DQN
import gymnasium as gym
import gym_examples
import numpy as np
import time
from tensorboardX import SummaryWriter

# Function to Write Evaluation Results
def write_to_file(word):
    with open("unmasked_dqn_evaluation_10_1.txt", "a") as file:
        file.write(word +"\n")

for i in range(2, 4):
    text = "Model: " + str(i)
    write_to_file(text)

    # Settting Model Files
    model_filename = "DQN EXTENSIVE TRAINING MODELS\\dqn_multiple_"+str(i)+".zip"
    logdir = "unmasked_dqn_"+ str(i) 
    writer = SummaryWriter(log_dir = logdir)

    # Load the DQN model from the saved file
    model = DQN.load(model_filename) # Use MaskedDQN for the Masked Version

    # Create your Tetris environment
    env = gym.make("gym_examples/Tetris-Binary-v0", width = 10, height = 10, reward_type = 7) # Can be changed to 10x20 Version

    # Define a function to record the environment
    def record_environment(env, model, num_timesteps=10000):
        t = 0
        episode = 0
        rewards = []
        actions = []
        lines = []
        lengths = []
        while(t <= num_timesteps):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            cleared_lines = 0
            ep_len = 0
            episode += 1
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                actions.append((action))
                if (i == 2) and (episode <= 5):
                    env.render()
                    time.sleep(2)
                episode_reward += reward
                cleared_lines += info["cleared_lines"]
                ep_len += 1
                t += 1
            rewards.append(episode_reward)
            lines.append(cleared_lines)
            lengths.append(ep_len)
            if (i == 2) and (episode <= 5):
                #print("Episode", episode, episode_reward, cleared_lines)
                list_as_string = ",".join(str(item) for item in actions)
                text = "Episode: " + str(episode) + " Reward: " +str(episode_reward) +" Cleared: " +str(cleared_lines) + " Actions: " + list_as_string + " Length: " +str(ep_len)
                write_to_file(text)
            writer.add_scalar("timesteps/episode_reward", episode_reward, t)
            writer.add_scalar("timesteps/episode_lines_cleared", (cleared_lines), t)
            writer.add_scalar("timesteps/episode_length", ep_len, t)
            writer.add_histogram("reward_distribution", rewards, t)
            writer.add_histogram("action_distribution", actions, t)
            
        text = "Average reward over 10000 timesteps " + str(np.mean(rewards))
        write_to_file(text)
        text = "Average lines over 10000 timesteps " + str(np.mean(lines))
        write_to_file(text)
        text = "Average length over 10000 timesteps " + str(np.mean(lengths))
        write_to_file(text)
        env.close()

    # Record the environment using the loaded model
    record_environment(env, model)

