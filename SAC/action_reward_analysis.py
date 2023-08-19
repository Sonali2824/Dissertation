# Import Statements
import random
import json
from AgentPreparation import AgentPreparation 
import torch
import numpy as np
import os

# Function to Write Reward Related Details
def write_to_file(data, filename):
    with open(filename, "a") as file:
        json.dump(data, file)
        file.write('\n')
        file.write('\n')

def action_reward_analysis(config):
    print("Action Reward Analysis")
    
    for i in range(1, 12): # Iterate over 11 Rewards
        config["seed"] = random.randint(1, 1000)
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
            
        for j in range(1, 2): # Iterate for 3 Training Cycles
            config["trial_number"] = i
            config["iteration_number"] = j
            config["reward"] = i
            
            
            agent = AgentPreparation(config)
            average_score, average_lines_cleared = agent.train_agent()
            print("Reward:", i)
            print("Average Score:", average_score)
            print("Average Lines Cleared:", average_lines_cleared)

            data = "Reward: " + str(i) + " Average Lines Cleared: " + str(average_lines_cleared) + " Average Score: " + str(average_score)
            write_to_file(data, "rewards_and_action.txt")