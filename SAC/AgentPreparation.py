# SAC Agent Implementation is adapted from https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch (Multiple Contributors)
# Import Statements
import gymnasium as gym
import gym_examples
from datetime import datetime
import os
import shutil
from tensorboardX import SummaryWriter
from SACDiscreteAgent import SAC_Discrete_Agent
import numpy as np
import torch
from utilities.Utility_Functions import create_actor_distribution

TRAINING_EPISODES_PER_EVAL_EPISODE = 10

class AgentPreparation(SAC_Discrete_Agent):
    def __init__(self, config):
        '''Initialising and Setting up Everything for Training the Agent'''

        # Storing Configuration Details
        self.config = config # Storing Miscellaneous Parameters
        self.hyperparameters = self.config["Hyperparameters"] # Storing Common Hyperparameters for Training the SAC Agent
        self.hyperparameters_actor = self.hyperparameters["Actor"] # Storing Actor Specific Hyperparameters
        self.hyperparameters_critic = self.hyperparameters["Critic"] # Storing Critic Specific Hyperparameters
        
        # Storing Tetris Enviornment Related Configuration Details
        self.environment = gym.make("gym_examples/Tetris-Binary-v0", width = self.config["board_width"], height = self.config["board_height"], reward_type = self.config["reward"]) # Tetris Enviornment
        self.action_size = int(self.environment.action_space.n)
        self.config["action_size"] = self.action_size     
        self.state_size =  self.environment.reset()[0].size
        self.action_types = "DISCRETE"

        # Setting Up Logging 
        gym.logger.set_level(40)  
        self.log_interval = self.config["log_interval"] # Logs Scalar Values Every 1000 Steps
        self.log_interval_distribution = self.config["log_interval_distribution"] # Logs Histograms Every 10000 Steps
        
        # Log Writer
        timenow = str(datetime.now())
        timenow = ' ' + timenow[0:19].replace(':', '_')
        writepath = 'SAC-D-masked-10-10-board-multiple-1/SAC-D-masked-10-10-multiple_{}'.format("tetris") + timenow + "_trial_" +str(self.config["trial_number"]) + "_itr_" + str(self.config["iteration_number"])
        if os.path.exists(writepath): shutil.rmtree(writepath)
        self.writer = SummaryWriter(log_dir=writepath)

        # Model Saving Interval
        self.save_model_interval = self.config["save_model_interval"]

        # Setting Up Enviornment Interaction Variables
        self.action_list = [] # To Monitor Actions Chosen       
        self.cleared_lines_list = [] # To Monitor the Lines Cleared
        self.episode_rewards_list = [] # To Monitor the Episode Rewards
        self.episode_lengths_list = [] # To Monitor the Episode Lengths  
        self.cleared_lines = 0
        self.total_episode_score_so_far = 0
        self.game_full_episode_scores = []
        self.episode_number = 0
        self.global_step_number = 0
        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]

        # Training Limitations - Episodic and Timesteps
        self.num_episodes = self.config["num_episodes_to_run"]
        self.timesteps = self.config["num_timesteps_to_run"]

        # Initialising the SAC Agent
        SAC_Discrete_Agent.__init__(self)

        print("Agent is Ready to Train")

    # Reset the Episodic Realted Variables at the End of an Episode
    def reset_episode_variables(self):
        self.environment.seed(self.config["seed"])
        self.state, _ = self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.cleared_lines = 0
        self.episode_rewards = []
        self.episode_actions = []

    def train_agent(self):
        # Code for training the agent

        while (self.episode_number < self.num_episodes) and (self.global_step_number < self.timesteps) :
            self.reset_episode_variables() # Reset Episodic Variables

            # Training Starts
            evaluation = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
            self.episode_step_number_val = 0
            self.environment._max_episode_steps = 30000 # Approximation Values
            episode_len = 0
            while not self.done:
                self.episode_step_number_val += 1
                state = self.state

                # Picking Action
                # Agent in Evaluation Mode and Training Mode Make Actor choose the Action
                if state is None: 
                    state = self.state
                state = torch.FloatTensor([state]).to(self.device)
                if len(state.shape) == 1: state = state.unsqueeze(0)
                action_probabilities = self.actor_local(state)
                max_probability_action = torch.argmax(action_probabilities, dim=-1)
                action_distribution = create_actor_distribution(self.action_types, action_probabilities, self.action_size)
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
                actor_picked_action = action[0]

                if evaluation: # Agent in Evaluation Mode
                    self.action = actor_picked_action
                elif self.global_step_number < self.hyperparameters["min_steps_before_learning"]: # Agent in Sampling Mode
                    if(self.config["isActionMasked"]): # Checks if Invalid Actions are to be Masked/Unmasked
                        _, masks, _ = self.environment.get_next_states()
                        self.action = self.environment.action_space.sample(masks) 
                        print("Picking random action ", self.action, masks)
                    else:
                        self.action = self.environment.action_space.sample() 
                        print("Picking random action ", self.action)
                else: # Agent in Training Mode
                    self.action = actor_picked_action
                
                self.action_list.append(self.action) # Monitoring Actions to Create Action Distribution
                
                # Perform the Action and Monitor State, Reward, Done, Info
                self.next_state, self.reward, self.done, _, self.info = self.environment.step(self.action)
                self.total_episode_score_so_far += self.reward
                self.cleared_lines += self.info["cleared_lines"]
                if self.hyperparameters["clip_rewards"]: # Clipping Rewards if clip_rewards = True
                    self.reward =  max(min(self.reward, 1.0), -1.0)
                
                # Check if Sufficient Samples are Collected, and Everything is Set for the Networks to Learn
                readyToTrainNetworks = self.global_step_number > self.hyperparameters["min_steps_before_learning"] and len(self.memory) > self.hyperparameters["batch_size"] and self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0
                
                # If readyToTrainNetwork --> True --> Train Actor and Critic Network
                if readyToTrainNetworks:
                    for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                        self.train_networks()

                # Adding Experiences to the Buffer
                if not evaluation: 
                    experience = self.state, self.action, self.reward, self.next_state, self.done
                    self.memory.add_experience(*experience)
                
                self.state = self.next_state

                self.global_step_number += 1 # Monitoring Timesteps
                episode_len += 1 # Monitoring Episode Length
            
            # Storing Episodic Data
            self.cleared_lines_list.append(self.cleared_lines)
            self.episode_rewards_list.append(self.total_episode_score_so_far)
            self.episode_lengths_list.append(episode_len)

            # Plots
            t = self.global_step_number 
            
            if t > 0 and self.global_step_number % self.log_interval == 0:
                # Reward
                self.writer.add_scalar('timesteps/episode_reward', self.total_episode_score_so_far, t)
                # Lines cleared
                self.writer.add_scalar('timesteps/episode_lines_cleared', self.cleared_lines, t)
                # Episode length
                self.writer.add_scalar('timesteps/episode_length', episode_len, t)

                # Average episodic lines cleared
                avg_lines_cleared = np.mean(self.cleared_lines_list)
                self.writer.add_scalar("timesteps-average/average_episode_lines_cleared", avg_lines_cleared, t)
                # Average episodic reward
                avg_ep_reward = np.mean(self.episode_rewards_list)
                self.writer.add_scalar("timesteps-average/average_episode_reward", avg_ep_reward, t)
                # Average episodic length
                avg_ep_len = np.mean(self.episode_lengths_list)
                self.writer.add_scalar("timesteps-average/average_episode_length", avg_ep_len, t)

                # Training Stability - Variance
                # if len(self.episode_rewards_list) % self.log_interval == 0:
                training_stability = np.var(self.episode_rewards_list[-100:])
                self.writer.add_scalar("timesteps/training_stability", training_stability, t)

            if t > 0 and len(self.episode_rewards_list) % self.log_interval_distribution == 0:
                rewards_hist = (self.episode_rewards_list)
                self.writer.add_histogram("reward_distribution", rewards_hist, t)

                # Logging action distribution data to TensorBoard every 100 episodes
                episode_actions = (self.action_list)
                self.writer.add_histogram("action_distribution", episode_actions, t)
            
            # End Plots
            if evaluation:
                print("----------------------------")
                print("Step number", self.global_step_number)
                print("Episode score {} ".format(self.total_episode_score_so_far))
                print("Episode number {} ".format(self.episode_number))
                print("Cleared Lines {} ".format(self.cleared_lines))
                print("Average Cleared Lines {}".format(np.mean(self.cleared_lines_list)))              
                print("----------------------------") 
            self.episode_number += 1
            self.game_full_episode_scores.append(self.total_episode_score_so_far) # Tracking All Episodic Scores
            
            # Storing the Model every 1000 steps
            if self.global_step_number % self.save_model_interval == 0:
                self.save_model()
        
        return np.mean(self.game_full_episode_scores), np.mean(self.cleared_lines_list)