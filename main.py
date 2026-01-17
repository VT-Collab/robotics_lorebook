import pybullet as p
import pybullet_data
import numpy as np
import os
import time
import config
from robot import Panda


# interpolated move to position
def move_to_pose(ee_position, ee_euler, n_steps=600):
    ee_euler[0] += np.pi
    state = panda.get_state()
    positions = np.linspace(state["ee-position"], ee_position, n_steps)
    eulers = np.linspace(state["ee-euler"], ee_euler, n_steps)
    for i in range(n_steps):
        panda.move_to_pose(positions[i], ee_quaternion=p.getQuaternionFromEuler(eulers[i]), positionGain=1.0)
        p.stepSimulation()
        time.sleep(config.control_dt)
    for i in range(200):
        panda.move_to_pose(ee_position, ee_quaternion=p.getQuaternionFromEuler(ee_euler), positionGain=1.0)
        p.stepSimulation()
        time.sleep(config.control_dt)

# open gripper
def open_gripper():
    for i in range(300):
        panda.open_gripper()
        p.stepSimulation()
        time.sleep(config.control_dt)

# open gripper
def close_gripper():
    for i in range(300):
        panda.close_gripper()
        p.stepSimulation()
        time.sleep(config.control_dt)

# terminate
def task_complete():
    p.disconnect()


# create simulation and place camera
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=config.cameraDistance, 
                                cameraYaw=config.cameraYaw,
                                cameraPitch=config.cameraPitch, 
                                cameraTargetPosition=config.cameraTargetPosition)

# load the objects
urdfRootPath = pybullet_data.getDataPath()
plane = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.625])
table = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.625])
cube = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"), basePosition=[0.6, -0.2, 0.3], useFixedBase=True, globalScaling=0.1)

# load the robot
panda = Panda(basePosition=config.baseStartPosition,
                baseOrientation=p.getQuaternionFromEuler(config.baseStartOrientationE),
                jointStartPositions=config.jointStartPositions)

# run sequence of position and gripper commands
move_to_pose([0.6, -0.4, 0.3], [0, -np.pi/2, 0])

move_to_pose([0.6, -0.4, 0.3], [np.pi/2, 0, -np.pi/2])