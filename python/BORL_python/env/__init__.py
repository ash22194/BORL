from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='env.frozenLake:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78 # optimum = .8196
)
