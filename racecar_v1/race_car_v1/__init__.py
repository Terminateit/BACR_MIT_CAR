from gym.envs.registration import register

register(
    id='racecar-v1',
    entry_point='race_car_v1.envs:CarRaceV1Env',
)