from gym.envs.registration import register

register(
    id='offload-autoscale-discrete-v0',
    entry_point='ppo.discrete.envs:OffloadAutoscaleDiscreteEnv',
)