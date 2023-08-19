# Import Statements
import json
import optuna
import torch
from AgentPreparation import AgentPreparation
from tuning_config import tuning_config
import random
import os
import torch
import numpy as np

trial_number = 1

# Function to Write Trail Related Details
def write_to_file(data, filename):
    with open(filename, "a") as file:
        json.dump(data, file)
        file.write('\n')
        file.write('\n')

# Objective Function to be Maximised
def objective(trial):
    global trial_number

    # Files to Write Trial Details
    filename = "Trial_reward_50_20_trials_re.txt"
    filename_score = "Scores_reward_50_20_trials_re.txt"    


    tuning_config["seed"] = 1
    '''Setting Seeds and Torch Parameters for the Required Enviornments for Reproducability'''

    # Setting Seed for Random and Numpy 
    random.seed(tuning_config["seed"])
    np.random.seed(tuning_config["seed"])

    # Setting Seed for OS
    os.environ["PYTHONHASHSEED"] = str(tuning_config["seed"])

    # Setting Seed for Torch
    torch.backends.cudnn.deterministic = True # Enforces Deterministic Behavior in CuDNN for Reproducability
    torch.backends.cudnn.benchmark = False # Disables CuDNN's Benchmarking Functionality to Choose the Best Algorithm for Reproducability
    torch.manual_seed(tuning_config["seed"])
    tuning_config["reward"] = trial.suggest_categorical('reward_type', [6, 7, 8, 9, 11])
    
    # Checking GPU Availability
    if torch.cuda.is_available():
        tuning_config["use_GPU"] = True
        write_to_file("cuda", filename)
    else:
        tuning_config["use_GPU"] = False
        write_to_file("cpu", filename)
    
    tuning_config["Hyperparameters"] = {            
            "clip_rewards": trial.suggest_categorical('clip_rewards', [True, False]),
            "batch_size": trial.suggest_categorical('batch_size', [64, 256, 640, 120, 360, 128, 512]),
            "automatically_tune_entropy_hyperparameter": True,
            "entropy_term_weight": None,
            "do_evaluation_iterations": True,
            "discount_rate": trial.suggest_categorical('discount_rate', [0.95, 0.99]),
            "update_every_n_steps": 1,
            "learning_updates_per_learning_session": 1,
            "min_steps_before_learning": trial.suggest_int('min_steps_before_learning', 200, 1000),
            "entropy_target": trial.suggest_categorical('entropy_target', [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98, 1]),

            "Actor": {
                "batch_norm": trial.suggest_categorical('batch_norm_actor', [False]),
                "linear_hidden_units": trial.suggest_categorical('linear_hidden_units_actor', [[256, 256], [32, 32], [50, 50, 50], [256, 200, 160, 110, 70], [200, 160, 110, 70], [71, 40]]),
                "hidden_activations":  trial.suggest_categorical('hidden_activations_actor',["relu", "leakyrelu", "sigmoid"]),
                "dropout": trial.suggest_categorical('dropout_actor', [0.3, 0.2]),
                "learning_rate": trial.suggest_categorical('learning_rate_actor', [0.0003, 0.001, 1e-5, 1e-4, 2e-6, 1e-3]),
                "final_layer_activation": "Softmax",
                "gradient_clipping_norm": trial.suggest_float('actor_gradient_clipping_norm_actor', 0.1, 10.0),
                "initialiser": trial.suggest_categorical('critic_initialiser_actor', ["Xavier", "He"])
            },

            "Critic": {
                "batch_norm": trial.suggest_categorical('batch_norm_critic', [False]),
                "linear_hidden_units": trial.suggest_categorical('linear_hidden_units_critic', [[256, 256], [32, 32], [50, 50, 50], [256, 200, 160, 110, 70], [200, 160, 110, 70], [71, 40]]),
                "hidden_activations":  trial.suggest_categorical('hidden_activations_critic',["relu", "leakyrelu", "sigmoid"]),
                "dropout": trial.suggest_categorical('dropout_critic', [0.3, 0.2]),
                "learning_rate": trial.suggest_categorical('learning_rate_critic', [0.0003, 0.001, 1e-5, 1e-4, 2e-6, 1e-3]),
                "buffer_size": trial.suggest_int('buffer_size_critic', 500000, 1000000),
                "tau": trial.suggest_categorical('tau_critic', [1, 0.005, 0.05]),
                "gradient_clipping_norm": trial.suggest_float('critic_gradient_clipping_norm_critic', 0.1, 10.0),
                "initialiser": trial.suggest_categorical('critic_initialiser_critic', ["Xavier", "He"]),
                "final_layer_activation": None
            }
        }
    

    first_part = "Trial:" + str(trial_number) + ": "
    config_str= first_part + str((tuning_config))
    write_to_file(config_str, filename)

    agent = AgentPreparation(tuning_config)
    average_score, average_lines_cleared = agent.train_agent()

    data = f"Lines Cleared {average_lines_cleared} for trial {trial_number}"
    write_to_file(data, filename_score)
    trial_number += 1

    return average_lines_cleared

def run_optuna():
    study = optuna.create_study(direction='maximize',
                                storage="sqlite:///db_20_50_re.sqlite3",  # Specifying the Tuning Data File
                                study_name="SAC-Masked-20_50_re-1")
    study.optimize(objective, n_trials=26) 

    print('Best trial:')
    best_trial = study.best_trial
    print('  Value: {}'.format(best_trial.value))
    print('  Params: ')
    for key, value in best_trial.params.items():
        print('    {}: {}'.format(key, value))

def hyperparameter_tuning():
    print("Hyperparameter Tuning")
    run_optuna()








