import gym
import racecar_v1
import racecar_v2

env = gym.make('racecar-v2')
env.reset()

done = False

while(done is False):
    env.render()
    observation, reward, done, _ =  env.step(env.action_space.sample())

env.close()