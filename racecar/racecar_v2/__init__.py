from gym.envs.registration import register

register(
    id='racecar-v2',
    entry_point='racecar_v2.envs:CarRaceEnv',
)