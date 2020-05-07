import gym
import racecar_v1
import racecar_v2

env = gym.make('racecar-v1')
env.reset(cameraStatus=True)

while(True):
    env.render()
    observation, reward, done, _ =  env.step(env.action_space.sample())

    if done:
        env.reset(cameraStatus=True)