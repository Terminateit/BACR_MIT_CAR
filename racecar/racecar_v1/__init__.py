from gym.envs.registration import register

register(
    id='racecar-v1',
    entry_point='racecar_v1.envs:CarRaceEnv',
)