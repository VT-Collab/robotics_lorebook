import pybullet as p
import pybullet_data
import numpy as np
import os


# class for the panda robot arm
# here is the link to the urdf files used for the panda:
# https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_data/franka_panda
class Panda():

    # the urdf for the panda robot has 11 joints
    # the first seven joints correspond to the joints of the robot arm, and the last two are for the gripper fingers
    # joints numbered 8 and 9 are meaningless, and are just used to build the robot model
    def __init__(self, basePosition, baseOrientation, jointStartPositions):
        self.urdfRootPath = pybullet_data.getDataPath()
        self.panda = p.loadURDF(os.path.join(self.urdfRootPath,"franka_panda/panda.urdf"), 
                                                basePosition=basePosition,
                                                baseOrientation=baseOrientation,
                                                useFixedBase=True)
        self.reset(jointStartPositions)

    # reset the panda robot to home_position
    # it is best only to do this at the start, while not running the simulation: resetJointState overrides all physics simulation.
    def reset(self, jointStartPositions):
        for idx in range(len(jointStartPositions)):
            p.resetJointState(self.panda, idx, jointStartPositions[idx])

    # get the robot's joint state and end-effector state
    def get_state(self):
        joint_values = p.getJointStates(self.panda, range(11))
        ee_values = p.getLinkState(self.panda, 11)
        state = {}
        state["joint-position"] = [item[0] for item in joint_values]
        state["joint-velocity"] = [item[1] for item in joint_values]
        state["joint-torque"] = [item[3] for item in joint_values]
        state["ee-position"] = ee_values[4]
        state["ee-quaternion"] = ee_values[5]
        state["ee-euler"] = p.getEulerFromQuaternion(state["ee-quaternion"])
        return state

    # close the robot's gripper
    # moves the fingers to positions [0.0, 0.0]
    # can tune the controller with "positionGains" as inputs to setJointMotorControlArray
    def close_gripper(self):
        positionGains = [0.01] * 2
        p.setJointMotorControlArray(self.panda, [9,10], p.POSITION_CONTROL, targetPositions=[0.0, 0.0], positionGains=positionGains)

    # open the robot's gripper
    # moves the fingers to positions [0.04, 0.04]
    # can tune the controller with "positionGains" as inputs to setJointMotorControlArray
    def open_gripper(self):
        positionGains = [0.01] * 2
        p.setJointMotorControlArray(self.panda, [9,10], p.POSITION_CONTROL, targetPositions=[0.04, 0.04], positionGains=positionGains)

    # inverse kinematics (IK) of the panda robot
    # computes the joint angles that makes the end-effector reach a given target position in Cartesian world space
    # optionally you can also specify the target orientation of the end effector using ee_quaternion
    # if ee_quaterion is set as None (i.e., not specified), pure position IK will be used
    def inverse_kinematics(self, ee_position, ee_quaternion):
        if ee_quaternion is None:
            return p.calculateInverseKinematics(self.panda, 11, list(ee_position))
        else:
            return p.calculateInverseKinematics(self.panda, 11, list(ee_position), list(ee_quaternion))

    # move the robot to a desired position
    # computes the joint angles needed to make the end-effector reach a given target position in Cartesian world space
    # optionally you can also specify the target orientation of the end effector using ee_quaternion
    # robot uses position control (a PD controller) to move to the target joint angles
    # can tune the controller with "positionGains" as inputs to setJointMotorControlArray
    def move_to_pose(self, ee_position, ee_rotz=None, ee_quaternion=None, positionGain=1.0):
        if ee_rotz is not None:
            ee_quaternion = p.getQuaternionFromEuler([np.pi, 0, ee_rotz])
        targetPositions = self.inverse_kinematics(ee_position, ee_quaternion)
        p.setJointMotorControlArray(self.panda, range(9), p.POSITION_CONTROL, targetPositions=targetPositions, positionGains=[positionGain]*9)