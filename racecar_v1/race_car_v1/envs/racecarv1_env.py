
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pybullet as p
import pybullet_data 
import os
import math
import time


class CarRaceV1Env(gym.Env):
    metadata = {'render.modes': ['human']}
 
    def __init__(self):
        self._observation = []
        self.action_space = spaces.Discrete(3)
        # velocity, steer, force
        self.observation_space = spaces.Box(np.array([0, 0, 50]),
                                    np.array([0, 0, 50])) 
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
        self.lastControlTime = time.time()
        self.lastLidarTime = time.time()
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]    
 
    def _step(self, action):
        self.set_actuator(action)
        p.stepSimulation()
        self.observation = self.compute_observation()
        reward = self.compute_reward()
        done = self.compute_done()
        self.envStepCounter += 1
        status = "Step " + str(self.envStepCounter) + " Reward " + '{0:.2f}'.format(reward)
        print(status)
        p.addUserDebugText(status, [0,-1,3], replaceItemUniqueId=1)
        return np.array(self.observation), reward, done, {}
 
    def _reset(self):
        self.vt = 0 # current velocity pf the wheels
        self.maxV = 120 # max lelocity, 235RPM = 24,609142453 rad/sec
        self.dw = 0 # current velocity pf the wheels
        self.maxW = 120 # max lelocity, 235RPM = 24,609142453

        self.envStepCounter = 0
        p.resetSimulation()
        p.setGravity(0, 0,-10) # m/s^2
        p.setTimeStep(1./120.) # the duration of a step in sec
        planeId = p.loadSDF("/race_car_v1/envs/meshes/barca_track.sdf", globalScaling=1)
        # planeId = p.loadURDF("plane.urdf")
        robotStartPos = [0,0,0.15]
        robotStartOrientation = p.getQuaternionFromEuler([0,0,self.np_random.uniform(low=-
        1.57, high=1.57)])
        path = os.path.abspath(os.path.dirname(__file__))
        self.car = p.loadURDF(os.path.join(path, "racecar.xml"), robotStartPos, robotStartOrientation)
        
        for wheel in range(p.getNumJoints(self.car)):
            # print("joint[",wheel,"]=", p.getJointInfo(self.car,wheel))
            p.setJointMotorControl2(self.car,wheel,p.VELOCITY_CONTROL,targetVelocity=0,force=0)
            p.getJointInfo(self.car,wheel)	
            # print(wheel)

        self.wheels = [8,15]

        c = p.createConstraint(self.car,9,self.car,11,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
        p.changeConstraint(c,gearRatio=1, maxForce=10000)

        c = p.createConstraint(self.car,10,self.car,13,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
        p.changeConstraint(c,gearRatio=-1, maxForce=10000)

        c = p.createConstraint(self.car,9,self.car,13,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
        p.changeConstraint(c,gearRatio=-1, maxForce=10000)

        c = p.createConstraint(self.car,16,self.car,18,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
        p.changeConstraint(c,gearRatio=1, maxForce=10000)


        c = p.createConstraint(self.car,16,self.car,19,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
        p.changeConstraint(c,gearRatio=-1, maxForce=10000)

        c = p.createConstraint(self.car,17,self.car,19,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
        p.changeConstraint(c,gearRatio=-1, maxForce=10000)

        c = p.createConstraint(self.car,1,self.car,18,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
        p.changeConstraint(c,gearRatio=-1, gearAuxLink = 15, maxForce=10000)
        c = p.createConstraint(self.car,3,self.car,19,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
        p.changeConstraint(c,gearRatio=-1, gearAuxLink = 15,maxForce=10000)

        self.steering = [0,2]

        self.hokuyo_joint=4
        self.zed_camera_joint = 5

        replaceLines=True
        self.numRays=100
        self.rayFrom=[]
        self.rayTo=[]
        self.rayIds=[]
        self.rayHitColor = [1,0,0]
        self.rayMissColor = [0,1,0]
        rayLen = 8
        rayStartLen=0.25
        for i in range (self.numRays):
            #self.rayFrom.append([0,0,0])
            self.rayFrom.append([rayStartLen*math.sin(-0.5*0.25*2.*math.pi+0.75*2.*math.pi*float(i)/self.numRays), rayStartLen*math.cos(-0.5*0.25*2.*math.pi+0.75*2.*math.pi*float(i)/self.numRays),0])
            self.rayTo.append([rayLen*math.sin(-0.5*0.25*2.*math.pi+0.75*2.*math.pi*float(i)/self.numRays), rayLen*math.cos(-0.5*0.25*2.*math.pi+0.75*2.*math.pi*float(i)/self.numRays),0])
            if (replaceLines):
                self.rayIds.append(p.addUserDebugLine(self.rayFrom[i], self.rayTo[i], self.rayMissColor,parentObjectUniqueId=self.car, parentLinkIndex=self.hokuyo_joint ))
            else:
                self.rayIds.append(-1)
        prevCarYaw = self.getCarYaw(self.car)
        self.observation = self.compute_observation()

        self.lastControlTime = time.time()
        self.lastLidarTime = time.time()
        return np.array(self.observation)
 
    def _render(self, mode='human', close=False):
        if (self.connectmode == 0):
            p.disconnect(self.physicsClient)
            # connect the graphic renderer
            self.physicsClient = p.connect(p.GUI)
            self.connectmode = 1
        pass

    def set_actuator(self, action):
        dv = 10
        vt = np.clip(self.vt + dv, -self.maxV, self.maxV)
        self.vt = vt
        for wheel in self.wheels:
            p.setJointMotorControl2(self.car,wheel,p.VELOCITY_CONTROL,targetVelocity=self.vt,force=50)
        for steer in self.steering:
            p.setJointMotorControl2(self.car,steer,p.POSITION_CONTROL,targetPosition=-self.vt)

    def getCarYaw(self,car):
        carPos,carOrn = p.getBasePositionAndOrientation(car)
        carEuler = p.getEulerFromQuaternion(carOrn)
        carYaw = carEuler[2]*360/(2.*math.pi)-90
        return carYaw


    def compute_observation(self):
        # camInfo = p.getDebugVisualizerCamera()

        self.nowControlTime = time.time()

        self.nowLidarTime = time.time()
        #lidar at 20Hz
        if (self.nowLidarTime-self.lastLidarTime>.3):
            print("Lidar!")
            numThreads=0
            results = p.rayTestBatch(self.rayFrom,self.rayTo,numThreads, parentObjectUniqueId=self.car, parentLinkIndex=self.hokuyo_joint)
            for i in range (self.numRays):
                hitObjectUid=results[i][0]
                hitFraction = results[i][2]
                hitPosition = results[i][3]
                if (hitFraction==1.):
                    p.addUserDebugLine(self.rayFrom[i],self.rayTo[i], self.rayMissColor,replaceItemUniqueId=self.rayIds[i],parentObjectUniqueId=self.car, parentLinkIndex=self.hokuyo_joint)
                else:
                    localHitTo = [self.rayFrom[i][0]+hitFraction*(self.rayTo[i][0]-self.rayFrom[i][0]),
                                                self.rayFrom[i][1]+hitFraction*(self.rayTo[i][1]-self.rayFrom[i][1]),
                                                self.rayFrom[i][2]+hitFraction*(self.rayTo[i][2]-self.rayFrom[i][2])]
                    p.addUserDebugLine(self.rayFrom[i],localHitTo, self.rayHitColor,replaceItemUniqueId=self.rayIds[i],parentObjectUniqueId=self.car, parentLinkIndex=self.hokuyo_joint)
            self.lastLidarTime = self.nowLidarTime
            
        #control at 100Hz
        carPos,carOrn = p.getBasePositionAndOrientation(self.car)


        carYaw = self.getCarYaw(self.car)
        cubeEuler = p.getEulerFromQuaternion(carOrn)
        linear, angular = p.getBaseVelocity(self.car)
        return (np.array([cubeEuler[0],angular[0],self.vt], dtype='float32'))

    def compute_reward(self):
        # receive a bonus of 1 for balancing and pay a small cost proportional to speed
        return 1.0 - abs(self.vt) * 0.05

    def compute_done(self):
        # episode ends when the barycentre of the robot is too low or after 500 steps
        cubePos, _ = p.getBasePositionAndOrientation(self.car)
        return cubePos[2] < 0.15 or self.envStepCounter >= 500  



    
