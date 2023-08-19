# Dissertation

## 📁 Decription of Folder and File Content

1. <a href="https://github.com/Sonali2824/Dissertation/tree/main/DQN%20COMPARITATIVE%20STUDY"> <b>DQN COMPARITIVE STUDY</b></a>
: Code used to extensively train a DQN Agent across 10 million timesteps and comapred with the SAC agent.
2. <a href="https://github.com/Sonali2824/Dissertation/tree/main/DQN%20HYPERPARAMETER%20TUNING"> <b>DQN HYPERPARAMETER TUNING</b></a>
: Code used to tune the DQN agent across 50 trials.

3. <a href="https://github.com/Sonali2824/Dissertation/tree/main/DQN%20REWARD%20AND%20ACTION%20REPRESENTATION%20ANALYSIS"> <b>DQN REWARD AND ACTION REPRESENTATION ANALYSIS</b></a>
: Code used to analyse the action and reward representations impact on the performance of the DQN agent.

4. <a href="https://github.com/Sonali2824/Dissertation/tree/main/RANDOM%20AGENT"> <b>RANDOM AGENT</b></a>
: Code used to generate masked and unmasked random agents.

5. <a href="https://github.com/Sonali2824/Dissertation/tree/main/gym-examples"> <b>gym-examples</b></a>
: Code to generate the Tetris enviornment.
    - Any change to the Tetris enviornment has to be made to file <a href="https://github.com/Sonali2824/Dissertation/blob/main/gym-examples/gym_examples/envs/tetris_high_state_space.py"> <b>TETRIS ENVIORNMENT .py FILE</b></a>
    - Gymnasium environment is coded in accordance to the example given on <a href="https://github.com/Farama-Foundation/gym-examples">Gym Examples</a>

6. To procure the logs related to Q-values the following file has to replace the original stable-baselines3 ```dqn.py``` file : <a href="https://github.com/Sonali2824/Dissertation/blob/main/stable_baselines_3_modified_dqn_code.py"><b>dqn.py</b></a>

7. <a href="https://github.com/Sonali2824/Dissertation/tree/main/DQN%20EVALUATION%20CODE"><b>DQN EVALUATION CODE</b></a>: Code used to evaluate the DQN agents.

8. <a href="https://github.com/Sonali2824/Dissertation/tree/main/SAC"><b>SAC CODE</b></a>: Code used to train, tune, and evaluate SAC agents.

## ⚙️ Running the code

```sh

# After any change made to the Tetris Enviornment follow these steps
cd gym-examples
pip install -e .

# Training/Evaluating Agent -- DQN
python <training/evaluating.py>

# SAC Agent
cd SAC
python main.py tuning # For Hyperparameter Tuning
python main.py evaluation # For Model Evaluation
python main.py analysis # For Action and Reward Representation Analysis
python main.py extensive # For Training across 10 Million Timesteps


# You can also  visualise the learning curves via TensorBoard
tensorboard --logdir <exp_name> # exp_name refers to the log directory
```
## 📖 Arguments and Hyperparameters
1. Extensive Training Config File: extensive_config.py
2. Tuning Config File: tuning_config.py
3. Analysis Config File: analysis_config.py
4. Evaluation Config File: Add as per Model

Updated config files as per requirement
``` sh
config = {
    "num_episodes_to_run": 5000, 
    "num_timesteps_to_run": 10000000,
    "board_width": 10,
    "board_height": 10,
    "isActionMasked": True,
    "reward": 6,
    "trial_number": 1,
    "iteration_number": 1,
    "log_interval": 1000,
    "log_interval_distribution": 10000,
    "save_model_interval": 1000,
    "Hyperparameters": {        
        "clip_rewards": True,
        "batch_size": 128,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "do_evaluation_iterations": True,
        "discount_rate": 0.99,
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "min_steps_before_learning": 314,
        "entropy_target": 0.3,
        "Actor": {
            "batch_norm": False,
            "linear_hidden_units": [256, 256],
            "hidden_activations": "leakyrelu",
            "dropout": 0.2,
            "learning_rate": 0.001,
            "final_layer_activation": "Softmax",
            "gradient_clipping_norm": 9.49191708591034,
            "initialiser": "He"
        },
        "Critic": {
            "batch_norm": False,
            "linear_hidden_units": [256, 256],
            "hidden_activations": "leakyrelu",
            "dropout": 0.2,
            "learning_rate": 0.001,
            "buffer_size": 938111,
            "tau": 0.05,
            "gradient_clipping_norm": 2.957642117580966,
            "initialiser": "Xavier",
            "final_layer_activation": None
        }
    }
}

```
