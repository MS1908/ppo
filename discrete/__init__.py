from gym.envs.registration import register

register(
    id='offload-autoscale-discrete',
    entry_point='discrete.envs:OffloadAutoscaleDiscreteEnv',
)