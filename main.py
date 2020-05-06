import gym
import racecar_v2

track_name = 'barca_track.sdf'
model_name = 'racecar_differential.urdf'
useRealTimeSim = 0

env = gym.make('racecar-v2')
env.reset(model_name, track_name, useRealTimeSim)

while(True):
    env.render()
    observation, step_reward, done, _ =  env.step(env.action_space.sample())
    
env.close()