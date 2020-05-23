# Thus environment based on
# https://github.com/erwincoumans/pybullet_robots/blob/master/f10_racecar.py

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import numpy as np
import csv
from PIL import Image

import pybullet as p
import pybullet_data 


IMAGE_W = 64
IMAGE_H = 64


class CarRaceEnv(gym.Env):

    def __init__(self):
        self.seed()
        self.viewer = None

        #p.connect(p.GUI)
        p.connect(p.DIRECT)
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

        # Observation space - camera snapshot
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

        # Joint numbers
        self.camera_joint = 5
        self.rearWheels = [8, 15]
        self.steerWheels = [0, 2]

        # Flag for the reset function
        self.world_does_exist = False

        # Counter to set the done flag
        self.stopThrehold = 100 # if the car is stuck more then this number of steps - reset


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def createWorld(self, model_name, track_name):
        self.resetCounter = 0

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
 
        # path = os.path.abspath(os.path.dirname(__file__))
        self.car = p.loadURDF(model_path, carPosition, carOrientation)

        for wheel in range(p.getNumJoints(self.car)):
            p.setJointMotorControl2(self.car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            p.getJointInfo(self.car, wheel)

        c = p.createConstraint(self.car, 9, self.car, 11, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
        p.changeConstraint(c, gearRatio=1, maxForce=10000)
        c = p.createConstraint(self.car, 10, self.car, 13, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
        p.changeConstraint(c, gearRatio=-1, maxForce=10000)
        c = p.createConstraint(self.car, 9, self.car, 13, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
        p.changeConstraint(c, gearRatio=-1, maxForce=10000)
        c = p.createConstraint(self.car, 16, self.car, 18, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
        p.changeConstraint(c, gearRatio=1, maxForce=10000)
        c = p.createConstraint(self.car, 16, self.car, 19, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
        p.changeConstraint(c, gearRatio=-1, maxForce=10000)
        c = p.createConstraint(self.car, 17, self.car, 19, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
        p.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = p.createConstraint(self.car, 1, self.car, 18, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
        p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
        c = p.createConstraint(self.car, 3, self.car, 19, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
        p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)



    def reset(self, model_name=None, track_name=None, cameraStatus=True, storeData=False):
        self.velocity = 0
        self.steeringAngle = 0
        self.force = 0

        self.storeData = storeData
        self.cameraStatus = cameraStatus
        self.cameraDirectory = 'snapshots'

        # reset counters
        self.stuckCounter = 0
        self.stepCounter = 0

        if self.world_does_exist is False:
            self.createWorld(model_name, track_name)
            self.world_does_exist = True
        else:
            self.resetCounter += 1
            carPosition = [0, 0, 0.15]
            carOrientation = p.getQuaternionFromEuler([0, 0, np.pi/3])
            p.resetBasePositionAndOrientation(self.car, carPosition, carOrientation)

        if cameraStatus is True:
            if self.storeData is True:
                # Create directory to store the snapshots
                if not os.path.exists(self.cameraDirectory):
                    os.makedirs(self.cameraDirectory)
                
                # Open the dataset csv file
                dataset_file = open('dataset.csv', 'a')
                self.dataset = csv.writer(dataset_file, lineterminator='\n')
                
                # Write csv header
                # columns = ['Current snapshot', 'Action space', 'Reward', 'Done', 'Next Snapshot']
                # self.dataset.writerow(columns)

            self.snapshot, self.snapshotPath, self.nextSnapshotPath = self.takeSnapshot()

        return self.snapshot


    def act(self):
        for wheel in self.rearWheels:
            p.setJointMotorControl2(self.car, wheel, p.VELOCITY_CONTROL, targetVelocity=self.velocity, force=self.force)
            
        for steer in self.steerWheels:
            p.setJointMotorControl2(self.car, steer, p.POSITION_CONTROL, targetPosition=-self.steeringAngle)


    def step(self, action):
        # Apply control action (120 Hz, each 1st step)
        if (self.stepCounter % 1 == 0):
            self.velocity = action[0]
            self.steeringAngle = action[1]
            self.force = action[2]
            self.act()

        # Update camera data (120 Hz, each 1st step)
        if self.cameraStatus is True:
            if (self.stepCounter % 1 == 0):
                self.snapshot, self.snapshotPath, self.nextSnapshotPath = self.takeSnapshot()

        p.stepSimulation()

        carVelocity = p.getBaseVelocity(self.car)[0] # get linear velocity only
        carSpeed = np.linalg.norm(carVelocity)
        reward = carSpeed*self.dt

        if carSpeed <= 0.3:
            self.stuckCounter += 1
            #print(self.stuckCounter, carSpeed)
            
        if self.stuckCounter >= self.stopThrehold:
            done = True
        else:
            done = False

        if self.storeData is True:
            datasetRow = [self.snapshotPath, action, reward, int(done), self.nextSnapshotPath]
            self.dataset.writerow(datasetRow)

        # Update step counter
        self.stepCounter += 1

        return self.snapshot, reward, done, {}
    

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
                                                                    projectionMatrix=projectionMatrix)
                                                                    #renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgbImg = Image.fromarray(rgbaImg).convert('RGB')
        
        savePath = os.path.join(self.cameraDirectory, 'snapshot_' + str(self.resetCounter) + '_' + str(self.stepCounter) + '.jpg')
        nextPath = os.path.join(self.cameraDirectory, 'snapshot_' + str(self.resetCounter) + '_' + str(self.stepCounter + 1) + '.jpg')
        if self.storeData is True:
            rgbImg.save(savePath, "JPEG")

        rgbImg = np.array(rgbImg)
        
        return rgbImg, savePath, nextPath


    def render(self):
        #if self.cameraStatus is True:
        #    from gym.envs.classic_control import rendering
        #    if self.viewer is None:
        #        self.viewer = rendering.SimpleImageViewer()
        #    self.viewer.imshow(self.snapshot)
        #    return self.snapshot
        #else:
        #    return 0
        return self.snapshot

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None