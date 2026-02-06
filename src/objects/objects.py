import pybullet as p
import pybullet_data
import os


# see list of pybullet objects here:
# https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_data
class PyBulletObject():

    def __init__(self, object_name, basePosition=[0.0, 0.0, 0.0], baseOrientation=[0.0, 0.0, 0.0, 1.0], globalScaling=1.0, useFixedBase=False):
        urdfRootPath = pybullet_data.getDataPath()
        self.object = p.loadURDF(os.path.join(urdfRootPath, object_name), basePosition=basePosition, baseOrientation=baseOrientation, globalScaling=globalScaling, useFixedBase=useFixedBase)

    def get_state(self):
        values = p.getBasePositionAndOrientation(self.object)
        state = {}
        state["position"] = values[0]
        state["quaternion"] = values[1]
        state["euler"] = p.getEulerFromQuaternion(state["quaternion"])
        return state


# see available simple objects in the folder:
# objects/simple_objects
class SimpleObject(PyBulletObject):

    def __init__(self, object_name, basePosition=[0.0, 0.0, 0.0], baseOrientation=[0.0, 0.0, 0.0, 1.0], globalScaling=1.0, useFixedBase=False):
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "simple_objects", object_name)        
        self.object = p.loadURDF(path, basePosition=basePosition, baseOrientation=baseOrientation, globalScaling=globalScaling, useFixedBase=useFixedBase)


# see available ycb objects in the folder:
# objects/ycb_objects
class YCBObject(PyBulletObject):

    def __init__(self, object_name, basePosition=[0.0, 0.0, 0.0], baseOrientation=[0.0, 0.0, 0.0, 1.0], globalScaling=0.08, useFixedBase=False):
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "ycb_objects", object_name)        
        self.object = p.loadURDF(path, basePosition=basePosition, baseOrientation=baseOrientation, globalScaling=globalScaling, useFixedBase=useFixedBase)


# see available collab objects in the folder:
# objects/collab_objects
class CollabObject():
    
    def __init__(self, object_name, basePosition=[0.0, 0.0, 0.0], baseOrientation=[0.0, 0.0, 0.0, 1.0], globalScaling=1.0, useFixedBase=True):
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "collab_objects", object_name)        
        self.object = p.loadURDF(path, basePosition=basePosition, baseOrientation=baseOrientation, globalScaling=globalScaling, useFixedBase=useFixedBase)

    # these objects have a revolute joint (e.g., door, microwave) or a prismatic joint (e.g., button, drawer, cabinet)
    # the "handle" refers to the link marked in red
    # the "joint_angle" is the position of the revolute or prismatic joint
    def get_state(self):
        values = p.getBasePositionAndOrientation(self.object)
        state = {}
        state["base_position"] = values[0]
        state["base_quaternion"] = values[1]
        state["base_euler"] = p.getEulerFromQuaternion(state["base_quaternion"])
        state["handle_position"] = p.getLinkState(self.object, 1)[0]
        state["handle_quaternion"] = p.getLinkState(self.object, 1)[1]
        state["handle_euler"] = p.getEulerFromQuaternion(state["handle_quaternion"])
        state["joint_angle"] = p.getJointState(self.object, 0)[0]
        return state