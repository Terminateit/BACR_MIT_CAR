import gym
import racecar_v2

track_name = 'barca_track.sdf'
model_name = 'racecar_differential.urdf'

env = gym.make('racecar-v2')
env.reset(model_name, track_name)
env.render(mode='human')

for _ in range(100000):
    for _ in range(1000000):
        a = 5**100
    env.render(mode='human')
    #env.step(env.action_space.sample()) # take a random action
#env.close()