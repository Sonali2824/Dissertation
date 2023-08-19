analysis_config = {
    "num_episodes_to_run": 5000, #50000000000000000000000000000000,
    "num_timesteps_to_run": 1000000000000000000,
    "board_width": 10,
    "board_height": 10,
    "isActionMasked": True,
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
        "min_steps_before_learning": 853,
        "entropy_target": 0.6,
        "Actor": {
            "batch_norm": False,
            "linear_hidden_units": [256, 256],
            "hidden_activations": "sigmoid",
            "dropout": 0.2,
            "learning_rate": 0.001,
            "final_layer_activation": "Softmax",
            "gradient_clipping_norm": 8.49952895643033,
            "initialiser": "He"
        },
        "Critic": {
            "batch_norm": False,
            "linear_hidden_units": [256, 256],
            "hidden_activations": "sigmoid",
            "dropout": 0.2,
            "learning_rate": 0.001,
            "buffer_size": 938111,
            "tau": 0.005,
            "gradient_clipping_norm": 7.622897181274258,
            "initialiser": "He",
            "final_layer_activation": None
        }
    }
}

