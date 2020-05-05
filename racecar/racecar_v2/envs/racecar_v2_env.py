# Thus environment based on
# https://github.com/erwincoumans/pybullet_robots/blob/master/f10_racecar.py

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import numpy as np
import pybullet as p
import pybullet_data 

STATE_W = 96
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 600
WINDOW_W = 1000
WINDOW_H = 800

FPS = 50

class CarRaceEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second' : FPS
    }
 
    def __init__(self):
        self.seed()
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF

        p.setGravity(0, 0, -9.81)
        useRealTimeSim = 0

        p.setTimeStep(1./120.)
        p.setRealTimeSimulation(useRealTimeSim)

    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, model_name, track_name):
        p.resetSimulation()

        model_path = os.path.join(os.path.dirname(__file__), 'f10_racecar', model_name)
        if not os.path.exists(model_path):
            raise IOError('Model file {} does not exist'.format(model_path))

        track_path = os.path.join(os.path.dirname(__file__), 'f10_racecar/meshes', track_name)
        if not os.path.exists(track_path):
            raise IOError('Track file {} does not exist'.format(track_path))
        
        self.track = p.loadSDF(track_path, globalScaling=1)

        self.seed()

        carPos = [0, 0, 0.15]
        carOrientation = p.getQuaternionFromEuler([0, 0, self.np_random.uniform(low=-1.57, high=1.57)])
        path = os.path.abspath(os.path.dirname(__file__))
        self.car = p.loadURDF(model_path, carPos, carOrientation) 

    def step(self, action):
        step_reward = 0
        done = False
        return self.state, step_reward, done, {}
    
    def snapshot(self):
        # Camera sensor number
        camera_joint = 5

        # Create camera projection matrix
        cameraInfo = p.getDebugVisualizerCamera()
        projectionMatrix = cameraInfo[3]
 
        # Get camera eye position (in Cartesian world coordinates) and orientation
        cameraState = p.getLinkState(self.car, camera_joint, computeForwardKinematics=True)       
        eyePosition = cameraState[0]
        eyeOrientation = cameraState[1]

        rotationMatrix = p.getMatrixFromQuaternion(eyeOrientation)
        rotationMatrix = np.array(rotationMatrix).reshape(3, 3)
        
        # Find cameraTarget - position of the target (focus) point, in Cartesian world coordinates
        initCameraVector = (1, 0, 0) # x-axis
        cameraVector = rotationMatrix.dot(initCameraVector)
        cameraTarget = eyePosition + 10 * cameraVector
        
        # Find cameraUpVector - up vector of the camera, in Cartesian world coordinates
        initUpVector = (0, 0, 1) # z-axis
        cameraUpVector = rotationMatrix.dot(initUpVector)

        viewMatrix = p.computeViewMatrix(eyePosition, cameraTarget, cameraUpVector)

        width, height, rgbImg, depthImg, segImg = p.getCameraImage(width=320,
                                                                   height=320,
                                                                   viewMatrix=viewMatrix,
                                                                   projectionMatrix=projectionMatrix)
        
        return rgbImg

    def render(self, mode='human'):
        assert mode in ['human', 'state_pixels', 'rgb_array']
        
        rgbImg = self.snapshot()
        
        pass






