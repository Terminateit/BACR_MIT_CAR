
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
        self.envStepCounter = 0
        p.resetSimulation()
        p.setGravity(0, 0,-10) # m/s^2
        p.setTimeStep(1./120.) # the duration of a step in sec
        print(os.path.abspath(os.getcwd()))
        planeId = p.loadSDF("/race_car_v1/envs/meshes/barca_track.sdf", globalScaling=1)
        # planeId = p.loadURDF("plane.urdf")
        robotStartPos = [0,0,.3]
        robotStartOrientation = p.getQuaternionFromEuler([self.np_random.uniform(low=-
        0.3, high=0.3),0,0])
        path = os.path.abspath(os.path.dirname(__file__))
        print(path)
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

        hokuyo_joint=4
        zed_camera_joint = 5

        replaceLines=True
        numRays=100
        rayFrom=[]
        rayTo=[]
        rayIds=[]
        rayHitColor = [1,0,0]
        rayMissColor = [0,1,0]
        rayLen = 8
        rayStartLen=0.25
        for i in range (numRays):
            #rayFrom.append([0,0,0])
            rayFrom.append([rayStartLen*math.sin(-0.5*0.25*2.*math.pi+0.75*2.*math.pi*float(i)/numRays), rayStartLen*math.cos(-0.5*0.25*2.*math.pi+0.75*2.*math.pi*float(i)/numRays),0])
            rayTo.append([rayLen*math.sin(-0.5*0.25*2.*math.pi+0.75*2.*math.pi*float(i)/numRays), rayLen*math.cos(-0.5*0.25*2.*math.pi+0.75*2.*math.pi*float(i)/numRays),0])
            if (replaceLines):
                rayIds.append(p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor,parentObjectUniqueId=self.car, parentLinkIndex=hokuyo_joint ))
            else:
                rayIds.append(-1)
        prevCarYaw = self.getCarYaw(self.car)
        self.observation = self.compute_observation()
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
            print(wheel)
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
        lastTime = time.time()
        lastControlTime = time.time()
        lastLidarTime = time.time()

        nowControlTime = time.time()
	
        nowLidarTime = time.time()
        #lidar at 20Hz
        if (nowLidarTime-lastLidarTime>.3):
            #print("Lidar!")
            numThreads=0
            results = p.rayTestBatch(rayFrom,rayTo,numThreads, parentObjectUniqueId=car, parentLinkIndex=hokuyo_joint)
            for i in range (numRays):
                hitObjectUid=results[i][0]
                hitFraction = results[i][2]
                hitPosition = results[i][3]
                if (hitFraction==1.):
                    p.addUserDebugLine(rayFrom[i],rayTo[i], rayMissColor,replaceItemUniqueId=rayIds[i],parentObjectUniqueId=car, parentLinkIndex=hokuyo_joint)
                else:
                    localHitTo = [rayFrom[i][0]+hitFraction*(rayTo[i][0]-rayFrom[i][0]),
                                                rayFrom[i][1]+hitFraction*(rayTo[i][1]-rayFrom[i][1]),
                                                rayFrom[i][2]+hitFraction*(rayTo[i][2]-rayFrom[i][2])]
                    p.addUserDebugLine(rayFrom[i],localHitTo, rayHitColor,replaceItemUniqueId=rayIds[i],parentObjectUniqueId=car, parentLinkIndex=hokuyo_joint)
            lastLidarTime = nowLidarTime
            
        #control at 100Hz
        carPos,carOrn = p.getBasePositionAndOrientation(self.car)

        # Keep the previous orientation of the camera set by the user.
        
        # yaw = camInfo[8]
        # pitch = camInfo[9]
        # distance = camInfo[10]
        # targetPos = camInfo[11]
        # camFwd = camInfo[5]
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



    
