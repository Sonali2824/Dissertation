from gymnasium.envs.registration import register

register(
    id="gym_examples/Tetris-Binary-v0",
    entry_point="gym_examples.envs:Tetris_Binary"
)

