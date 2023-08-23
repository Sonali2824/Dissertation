# Import Statements
import sys
from action_reward_analysis import action_reward_analysis
from evaluation import evaluation
from extensive_training import extensive_training
from hyperparameter_tuning import hyperparameter_tuning
from extensive_config import config
from analysis_config import analysis_config
from evaluation_extensive import evaluation_extensive


option = sys.argv[1] # Procuring Action to be Performed on the SAC Agent
   
# Switching into the Mode of Operation 
if option == "extensive": # Extensive Training of the SAC Agent across 10 Million Timesteps
    extensive_training(config)

elif option == "tuning": # Perform Hyperparameter Tuning
    hyperparameter_tuning()

elif option == "analysis": # Perform Analysis of Reward and Action Representations
    action_reward_analysis(analysis_config)

elif option == 'evaluation': # Perform Evaluation of SAC Agents
    evaluation(config)

elif option == 'evaluation_extensive': # Perform Evaluation of Extensively Trained SAC Agents
    evaluation_extensive(config)

    