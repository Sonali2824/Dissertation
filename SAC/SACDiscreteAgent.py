# SAC Agent Implementation is adapted from https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch (Multiple Contributors)
# Import Statements
import torch
from torch.optim import Adam
import numpy as np
from nn_builder.pytorch.NN import NN
from utilities.data_structures.Replay_Buffer import Replay_Buffer
import os
import torch.nn.functional as F
from utilities.Utility_Functions import create_actor_distribution

class SAC_Discrete_Agent(object):
    def __init__(self):
        '''Initialising Parameters and Networks for the SAC-D Agent'''

        # Setting Up Device
        self.device = "cuda:0" if self.config["use_GPU"] else "cpu" # Checks if GPU/CPU is to be Used

        # Entropy Tuning as per Paper: https://arxiv.org/pdf/1812.05905.pdf
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            self.target_entropy = -np.log((1.0 / self.action_size)) * self.hyperparameters["entropy_target"] # "entropy_target" is also tuned in this implementation as the given 0.98 doesn't work well with Tetris
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]

        # Setting Up Networks as Mentioned in https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
        # Creating Critic - 1 
        self.critic_local_1 = NN(input_dim=self.state_size, layers_info=self.hyperparameters_critic["linear_hidden_units"] + [self.action_size],
                  output_activation=self.hyperparameters_critic["final_layer_activation"],
                  batch_norm=self.hyperparameters_critic["batch_norm"], dropout=self.hyperparameters_critic["dropout"],
                  hidden_activations=self.hyperparameters_critic["hidden_activations"], initialiser=self.hyperparameters_critic["initialiser"],
                  columns_of_data_to_be_embedded=[], embedding_dimensions=[], y_range=(),
                  random_seed=self.config["seed"]).to(self.device) # Except for columns_of_data_to_be_embedded=[], embedding_dimensions=[], y_range=(): all other parameters were tuned     
        
        # Creating Critic - 2 To Prevent Overestimation of Q-Values
        self.critic_local_2 = NN(input_dim=self.state_size, layers_info=self.hyperparameters_critic["linear_hidden_units"] + [self.action_size],
                  output_activation=self.hyperparameters_critic["final_layer_activation"],
                  batch_norm=self.hyperparameters_critic["batch_norm"], dropout=self.hyperparameters_critic["dropout"],
                  hidden_activations=self.hyperparameters_critic["hidden_activations"], initialiser=self.hyperparameters_critic["initialiser"],
                  columns_of_data_to_be_embedded=[], embedding_dimensions=[], y_range=(),
                  random_seed=self.config["seed"] + 1).to(self.device)
        
        # Creating Optimisers for the Critics
        self.critic_optimiser_1 = torch.optim.Adam(self.critic_local_1.parameters(), lr=self.hyperparameters_critic["learning_rate"], eps=1e-4)
        self.critic_optimiser_2 = torch.optim.Adam(self.critic_local_2.parameters(), lr=self.hyperparameters_critic["learning_rate"], eps=1e-4)

        # Creating 2 Critic Targets
        self.critic_target_1 = NN(input_dim=self.state_size, layers_info=self.hyperparameters_critic["linear_hidden_units"] + [self.action_size],
                  output_activation=self.hyperparameters_critic["final_layer_activation"],
                  batch_norm=self.hyperparameters_critic["batch_norm"], dropout=self.hyperparameters_critic["dropout"],
                  hidden_activations=self.hyperparameters_critic["hidden_activations"], initialiser=self.hyperparameters_critic["initialiser"],
                  columns_of_data_to_be_embedded=[], embedding_dimensions=[], y_range=(),
                  random_seed=self.config["seed"]).to(self.device)
        
        self.critic_target_2 = NN(input_dim=self.state_size, layers_info=self.hyperparameters_critic["linear_hidden_units"] + [self.action_size],
                  output_activation=self.hyperparameters_critic["final_layer_activation"],
                  batch_norm=self.hyperparameters_critic["batch_norm"], dropout=self.hyperparameters_critic["dropout"],
                  hidden_activations=self.hyperparameters_critic["hidden_activations"], initialiser=self.hyperparameters_critic["initialiser"],
                  columns_of_data_to_be_embedded=[], embedding_dimensions=[], y_range=(),
                  random_seed=self.config["seed"]).to(self.device)
        
        # Creating Actor Network
        self.actor_local = NN(input_dim=self.state_size, layers_info=self.hyperparameters_actor["linear_hidden_units"] + [self.action_size],
                  output_activation=self.hyperparameters_actor["final_layer_activation"],
                  batch_norm=self.hyperparameters_actor["batch_norm"], dropout=self.hyperparameters_actor["dropout"],
                  hidden_activations=self.hyperparameters_actor["hidden_activations"], initialiser=self.hyperparameters_actor["initialiser"],
                  columns_of_data_to_be_embedded=[], embedding_dimensions=[], y_range=(),
                  random_seed=self.config["seed"]).to(self.device)

        # Creating Actor Optimiser
        self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=self.hyperparameters_actor["learning_rate"], eps=1e-4)

        # Initialising the Replay Buffer implemented in https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/utilities/data_structures/Replay_Buffer.py
        self.memory = Replay_Buffer(self.hyperparameters_critic["buffer_size"], self.hyperparameters["batch_size"], self.config["seed"], device=self.device)

        # Synchronising the Weights of Target NNs with the Weights of the Corresponding Local NNs to Stabalise Training
        self.copy_model_over(self.critic_local_1, self.critic_target_1)
        self.copy_model_over(self.critic_local_2, self.critic_target_2)
        
        print("SAC-D Initialised")
    
    @staticmethod # Model Copying Process Function from: https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
    def copy_model_over(from_model, to_model):
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())

    # Optimisation Process Function from: https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
    def optimisation(self, optimiser, network, loss, clipping_norm=None, retain_graph=False):
        if not isinstance(network, list): network = [network]
        optimiser.zero_grad() #reset gradients to 0
        loss.backward(retain_graph=retain_graph) #this calculates the gradients
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm) #clip gradients to help stabilise training
        optimiser.step() #this applies the gradients

    # Soft Update Process to Stabalise Training, Function from: https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
    def updation(self, local_model, target_model, tau):        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    # Function to Train Actor, and Critic Networks, and Temperature Parameter
    def train_networks(self):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample()

        # Calculating Critic Losses
        with torch.no_grad():
            # Next Q Value is Calculated as per the Formula Provided in: https://arxiv.org/pdf/1910.07207v2.pdf
            action_probabilities = self.actor_local(next_state_batch)
            # Dealing with log 0
            z = action_probabilities == 0.0
            z = z.float() * 1e-8
            log_action_probabilities = torch.log(action_probabilities + z)
            qf1_next_target = self.critic_target_1(next_state_batch)
            qf2_next_target = self.critic_target_2(next_state_batch)
            min_qf_next_target = action_probabilities * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (min_qf_next_target)

        qf1 = self.critic_local_1(state_batch).gather(1, action_batch.long())
        qf2 = self.critic_local_2(state_batch).gather(1, action_batch.long())
        q_values_1 = self.critic_local_1(state_batch)
        q_values_2 = self.critic_local_2(state_batch)

        # Log Q Values
        if self.global_step_number > 0 and self.global_step_number % self.log_interval == 0:
            # Log the Q-values as scalar values
            for action_idx in range(40):
                avg_q_value = torch.mean(q_values_1[:, action_idx])
                self.writer.add_scalar(f"Q-Values-1/Action_{action_idx}/Average_Q_Value", avg_q_value, self.global_step_number)
                self.writer.add_scalar(f"Q-Values-1/Action_{action_idx}/Max_Q_Value", torch.max(q_values_1[:, action_idx]), self.global_step_number)
                self.writer.add_scalar(f"Q-Values-1/Action_{action_idx}/Min_Q_Value", torch.min(q_values_1[:, action_idx]), self.global_step_number)

            # Log the Q-values as scalar values
            for action_idx in range(40):
                avg_q_value_2 = torch.mean(q_values_2[:, action_idx])
                self.writer.add_scalar(f"Q-Values-2/Action_{action_idx}/Average_Q_Value", avg_q_value_2, self.global_step_number)
                self.writer.add_scalar(f"Q-Values-2/Action_{action_idx}/Max_Q_Value", torch.max(q_values_2[:, action_idx]), self.global_step_number)
                self.writer.add_scalar(f"Q-Values-2/Action_{action_idx}/Min_Q_Value", torch.min(q_values_2[:, action_idx]), self.global_step_number)

        if self.global_step_number > 0 and len(self.episode_rewards_list) % self.log_interval_distribution == 0:
            # Log the Q-values as histograms
            for action_idx in range(40):
                self.writer.add_histogram(f"Q-Values-1/Action_{action_idx}/Histogram", q_values_1[:, action_idx], self.global_step_number)
            # Log the Q-values as histograms
            for action_idx in range(40):
                self.writer.add_histogram(f"Q-Values-2/Action_{action_idx}/Histogram", q_values_2[:, action_idx], self.global_step_number)
        
        # Computing MSE Losses
        critic_1_loss = F.mse_loss(qf1, next_q_value)
        critic_2_loss = F.mse_loss(qf2, next_q_value)

        # Optimisation and Updation of Critic Parameters
        # Critic Local - 1
        self.optimisation(self.critic_optimiser_1, self.critic_local_1, critic_1_loss, self.hyperparameters_critic["gradient_clipping_norm"])
        self.updation(self.critic_local_1, self.critic_target_1, self.hyperparameters_critic["tau"])
        
        # Critic Local - 2
        self.optimisation(self.critic_optimiser_2, self.critic_local_2, critic_2_loss, self.hyperparameters_critic["gradient_clipping_norm"])
        self.updation(self.critic_local_2, self.critic_target_2, self.hyperparameters_critic["tau"])
        
        # Logging Critic Losses
        if self.global_step_number > 0 and self.global_step_number % self.log_interval == 0:
            self.writer.add_scalar('loss/critic_1_loss', critic_1_loss, self.global_step_number)
            self.writer.add_scalar('loss/critic_2_loss', critic_2_loss, self.global_step_number)

        # Calculate Actor Loss as per the Formula Provided in: https://arxiv.org/pdf/1910.07207v2.pdf
        action_probabilities = self.actor_local(state_batch)
        # Dealing with log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)

        qf1_pi = self.critic_local_1(state_batch)
        qf2_pi = self.critic_local_2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha * log_action_probabilities - min_qf_pi
        actor_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        action_probabilities_1 = torch.exp(log_action_probabilities)

        # Calculating the entropy as per the Formula Provided in: https://arxiv.org/pdf/1910.07207v2.pdf
        entropy = -torch.sum(action_probabilities_1 * torch.log(action_probabilities_1))
        # Logging Entropy to Compare with DQN Exploration Rate
        if self.global_step_number > 0 and self.global_step_number % self.log_interval == 0:
            self.writer.add_scalar("rollout/exploration_rate", entropy, self.global_step_number)

        # Logging Actor Loss
        if self.global_step_number > 0 and self.global_step_number % self.log_interval == 0:
            self.writer.add_scalar('loss/actor_loss', actor_loss, self.global_step_number)

        # Computing alpha Loss
        if self.automatic_entropy_tuning: 
            alpha_loss = -(self.log_alpha * (log_action_probabilities + self.target_entropy).detach()).mean()
        else: alpha_loss = None

        # Log Alpha Loss
        if self.global_step_number > 0 and self.global_step_number % self.log_interval == 0:
            self.writer.add_scalar('loss/alpha_loss', alpha_loss, self.global_step_number)
        
        # Optimisation of Actor Parameters
        self.optimisation(self.actor_optimiser, self.actor_local, actor_loss, self.hyperparameters_actor["gradient_clipping_norm"])
        
        if alpha_loss is not None:
            self.optimisation(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()
        

    # Function to Save Models Locally
    def save_model(self):
        # File paths for saving
        critic_local_path = "{}_{}_critic_local_network_10_200.pt".format(str(self.config["trial_number"]), str(self.config["iteration_number"]))
        critic_local_2_path = "{}_{}_critic_local_2_network_10_200.pt".format(str(self.config["trial_number"]), str(self.config["iteration_number"]))
        critic_target_path = "{}_{}_critic_target_network_10_200.pt".format(str(self.config["trial_number"]), str(self.config["iteration_number"]))
        critic_target_2_path = "{}_{}_critic_target_2_local_network_10_200.pt".format(str(self.config["trial_number"]), str(self.config["iteration_number"]))
        actor_local_path = "{}_{}_actor_local_network_10_200.pt".format(str(self.config["trial_number"]), str(self.config["iteration_number"]))
        
        # Delete existing files if they exist
        for file_path in [critic_local_path, critic_local_2_path, critic_target_path, critic_target_2_path, actor_local_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted existing file: {file_path}")
        
        # Save new versions of the model
        torch.save(self.critic_local_1.state_dict(), critic_local_path)
        torch.save(self.critic_local_2.state_dict(), critic_local_2_path)
        torch.save(self.critic_target_1.state_dict(), critic_target_path)
        torch.save(self.critic_target_2.state_dict(), critic_target_2_path)
        torch.save(self.actor_local.state_dict(), actor_local_path)
        print("New model versions saved successfully.")

       

