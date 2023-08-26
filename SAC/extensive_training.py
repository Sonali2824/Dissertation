# Import Statements
from AgentPreparation import AgentPreparation
import torch
import os
import numpy as np
import random

def extensive_training(config):
    print("Extensive Training", config)
    config["seed"] = 1 #random.randint(1, 1000) # Mention a Random or Assigned Seed

    '''Setting Seeds and Torch Parameters for the Required Environments for Reproducability'''

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

    agent = AgentPreparation(config)
    average_score, average_lines_cleared = agent.train_agent()
    print("Average Score:", average_score)
    print("Average Lines Cleared:", average_lines_cleared)
