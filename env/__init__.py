from gym.envs.registration import register

register(
    id='CartPoleMy-v0',
    entry_point='env.cartpole:CartPoleEnv',
)