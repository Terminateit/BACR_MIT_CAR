# Thus environment based on
# https://github.com/erwincoumans/pybullet_robots/blob/master/f10_racecar.py

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import time
import numpy as np
from PIL import Image

import pybullet as p
import pybullet_data 


IMAGE_W = 320
IMAGE_H = 320


class CarRaceEnv(gym.Env):

    def __init__(self):
        self.seed()
        self.viewer = None

        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.setGravity(0, 0, -9.81)

        self.dt = 1./120.
        p.setTimeStep(self.dt)
        useRealTimeSim = 0
        p.setRealTimeSimulation(useRealTimeSim)
        
        self.velocityBound = 50 # 235RPM = 24,609142453 rad/sec
        self.steerBound = 1
        
        # Action space - [velocity, steering angle, force]
        self.action_space = spaces.Box(np.array([-self.velocityBound, -self.steerBound, 0]), 
                                        np.array([self.velocityBound, self.steerBound, 50]), dtype = np.float64)

        # Observation space - image from the camera on the car
        self.numRays = 100 # number of lidar rays
        self.observation_space = spaces.Box(low=np.zeros(self.numRays), high=np.ones(self.numRays), dtype=np.uint8)

        # Joint numbers
        self.lidar_joint = 4
        self.camera_joint = 5
        self.rearWheels = [8, 15]
        self.steerWheels = [0, 2]

        # Flag for the reset function
        self.world_does_exist = False


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def createWorld(self, model_name, track_name):

        if model_name is None:
            model_name = 'racecar_differential.urdf'

        if track_name is None:
            track_name = 'barca_track.sdf'

        model_path = os.path.join(os.path.dirname(__file__), 'f10_racecar', model_name)
        if not os.path.exists(model_path):
            raise IOError('Model file {} does not exist'.format(model_path))
        
        track_path = os.path.join(os.path.dirname(__file__), 'f10_racecar/meshes', track_name)
        if not os.path.exists(track_path):
            raise IOError('Track file {} does not exist'.format(track_path))
        
        self.track = p.loadSDF(track_path, globalScaling=1)

        carPosition = [0, 0, 0.15]
        carOrientation = p.getQuaternionFromEuler([0, 0, np.pi/3])
 
        path = os.path.abspath(os.path.dirname(__file__))
        self.car = p.loadURDF(model_path, carPosition, carOrientation)

        for wheel in range(p.getNumJoints(self.car)):
            p.setJointMotorControl2(self.car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            p.getJointInfo(self.car, wheel)	

        c = p.createConstraint(self.car, 9, self.car, 11, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
        p.changeConstraint(c,gearRatio=1, maxForce=10000)
        c = p.createConstraint(self.car, 10, self.car, 13, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
        p.changeConstraint(c,gearRatio=-1, maxForce=10000)
        c = p.createConstraint(self.car, 9, self.car, 13, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
        p.changeConstraint(c,gearRatio=-1, maxForce=10000)
        c = p.createConstraint(self.car, 16, self.car, 18, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
        p.changeConstraint(c,gearRatio=1, maxForce=10000)
        c = p.createConstraint(self.car, 16, self.car, 19, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
        p.changeConstraint(c,gearRatio=-1, maxForce=10000)
        c = p.createConstraint(self.car, 17, self.car, 19, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
        p.changeConstraint(c,gearRatio=-1, maxForce=10000)

        c = p.createConstraint(self.car, 1, self.car, 18, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
        p.changeConstraint(c,gearRatio=-1, gearAuxLink = 15, maxForce=10000)
        c = p.createConstraint(self.car, 3, self.car, 19, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
        p.changeConstraint(c,gearRatio=-1, gearAuxLink = 15, maxForce=10000)

        self.rayFrom = []
        self.rayTo = []
        self.rayIds = []
        self.rayHitColor = [1, 0, 0]
        self.rayMissColor = [0, 1, 0]
        rayLen = 4
        rayStartLen = 0.25
        for i in range (self.numRays):
            self.rayFrom.append([rayStartLen*np.sin(-0.5*0.25*2.*np.pi+0.75*2.*np.pi*float(i)/self.numRays),
                                 rayStartLen*np.cos(-0.5*0.25*2.*np.pi+0.75*2.*np.pi*float(i)/self.numRays),
                                 0
                                 ])

            self.rayTo.append([rayLen*np.sin(-0.5*0.25*2.*np.pi+0.75*2.*np.pi*float(i)/self.numRays),
                               rayLen*np.cos(-0.5*0.25*2.*np.pi+0.75*2.*np.pi*float(i)/self.numRays),
                               0
                               ])

            self.rayIds.append(p.addUserDebugLine(self.rayFrom[i], self.rayTo[i], self.rayMissColor, parentObjectUniqueId=self.car, parentLinkIndex=self.lidar_joint))


    def reset(self, model_name=None, track_name=None, cameraStatus=False):
        self.velocity = 0
        self.steeringAngle = 0
        self.force = 0

        self.cameraStatus = cameraStatus

        self.stopThrehold = 100 # if the car is stuck more then this number of steps - reset
        self.stuckCounter = 0 # reset stuck counter


        if self.world_does_exist is False:
            self.createWorld(model_name, track_name)
            self.world_does_exist = True
        else:
            carPosition = [0, 0, 0.15]
            carOrientation = p.getQuaternionFromEuler([0, 0, np.pi/3])
            p.resetBasePositionAndOrientation(self.car, carPosition, carOrientation)

        if self.cameraStatus is True:
            self.snapshot = self.takeSnapshot()

        self.observation = np.zeros(self.numRays, dtype=np.int)

        self.lastControlTime = 0
        self.lastLidarTime = 0
        self.lastCameraTime = 0
        self.currentTime = 0

        return self.step(None)[0]


    def getCarYaw(self):
        carPosition, carOrientation = p.getBasePositionAndOrientation(self.car)
        carEuler = p.getEulerFromQuaternion(carOrientation)
        carYaw = (carEuler[2]*360)/(2.*np.pi) - 90
        return carYaw


    def act(self):
        self.lastControlTime = self.currentTime
        for wheel in self.rearWheels:
            p.setJointMotorControl2(self.car, wheel, p.VELOCITY_CONTROL, targetVelocity=self.velocity, force=self.force)
            
        for steer in self.steerWheels:
            p.setJointMotorControl2(self.car, steer, p.POSITION_CONTROL, targetPosition=-self.steeringAngle)


    def step(self, action):
        # Apply control action (120 Hz)
        if (self.currentTime - self.lastControlTime > 1/120.):
            self.velocity = action[0]
            self.steeringAngle = action[1]
            self.force = action[2]
            self.act()

        if self.cameraStatus is True:
            # Update camera data (1 Hz)
            if (self.currentTime - self.lastCameraTime > 1.0):
                self.lastCameraTime = self.currentTime

                self.snapshot = self.takeSnapshot()
            
        # Update lidar data (120 Hz)
        if (self.currentTime - self.lastLidarTime > 1/120.):
            self.lastLidarTime = self.currentTime

            hitFractions = np.zeros(self.numRays, dtype=np.int8)

            numThreads = 0
            results = p.rayTestBatch(self.rayFrom, self.rayTo, numThreads, parentObjectUniqueId=self.car, parentLinkIndex=self.lidar_joint)
            for i in range (self.numRays):
                hitObjectUid = results[i][0]
                hitFraction = results[i][2]
                hitPosition = results[i][3]
                
                hitFractions[i] = hitFraction

                if (hitFraction == 1.):
                    p.addUserDebugLine(self.rayFrom[i], self.rayTo[i], self.rayMissColor, replaceItemUniqueId=self.rayIds[i], parentObjectUniqueId=self.car, parentLinkIndex=self.lidar_joint)
                else:
                    localHitTo = [self.rayFrom[i][0] + hitFraction*(self.rayTo[i][0] - self.rayFrom[i][0]),
                                  self.rayFrom[i][1] + hitFraction*(self.rayTo[i][1] - self.rayFrom[i][1]),
                                  self.rayFrom[i][2] + hitFraction*(self.rayTo[i][2] - self.rayFrom[i][2])]
                    p.addUserDebugLine(self.rayFrom[i], localHitTo, self.rayHitColor,replaceItemUniqueId=self.rayIds[i],parentObjectUniqueId=self.car, parentLinkIndex=self.lidar_joint)
            
            self.observation = hitFractions
            #print(self.observation)

        p.stepSimulation()

        self.currentTime += self.dt

        carVelocity = p.getBaseVelocity(self.car)[0] # get linear velocity only
        carSpeed = np.linalg.norm(carVelocity)
        reward = carSpeed*self.dt

        if carSpeed <= 0.2:
            self.stuckCounter += 1
            #print(self.stuckCounter, carSpeed)
            
        if self.stuckCounter >= self.stopThrehold:
            done = True
        else:
            done = False

        return self.observation, reward, done, {}
    

    def takeSnapshot(self):
        # Create camera projection matrix
        fov = 60
        aspect = 1.0
        nearPlane = 0.01
        farPlane = 100
        projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
 
        # Get camera eye position and orientation in Cartesian world coordinates
        cameraState = p.getLinkState(self.car, self.camera_joint, computeForwardKinematics=True)       
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

        width, height, rgbaImg, depthImg, segImg = p.getCameraImage(width=IMAGE_W,
                                                                    height=IMAGE_H,
                                                                    viewMatrix=viewMatrix,
                                                                    projectionMatrix=projectionMatrix,
                                                                    renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgbImg = Image.fromarray(rgbaImg).convert('RGB')
        rgbImg = np.array(rgbImg)
        
        return rgbImg


    def render(self):
        if self.cameraStatus is True:
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(self.snapshot)
            return self.viewer.isopen
        else:
            return False


    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None