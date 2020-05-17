import gym
import numpy as np
import matplotlib.pyplot as plt

import racecar_v1
import racecar_v2

env = gym.make('racecar-v2')
env.reset(cameraStatus=True, storeData=False)


fig = plt.figure()
ax = fig.gca()
snapshot = np.zeros((64, 64, 3), dtype=np.uint8)
render_object = ax.imshow(snapshot)

while(True):

    snapshot = env.render()
    render_object.set_data(snapshot)
    plt.draw()
    plt.pause(1e-6)


    observation, reward, done, _ =  env.step(env.action_space.sample())

    if done:
        env.reset(cameraStatus=True, storeData=False)
