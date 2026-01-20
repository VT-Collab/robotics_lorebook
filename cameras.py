import pybullet as p
import numpy as np


# class for an external camera with a fixed base
# this camera does not move during the simulation
class ExternalCamera():

	def __init__(self, cameraDistance=1.0, cameraYaw=45, cameraPitch=-30, cameraRoll=0, cameraTargetPosition=[0,0,0], cameraWidth=256, cameraHeight=256):

		self.cameraDistance = cameraDistance
		self.cameraYaw = cameraYaw
		self.cameraPitch = cameraPitch
		self.cameraRoll = cameraRoll
		self.cameraTargetPosition = cameraTargetPosition
		self.cameraWidth = cameraWidth
		self.cameraHeight = cameraHeight

		self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
														aspect=self.cameraWidth / self.cameraHeight,
														nearVal=0.01,
														farVal=2.0)

		self.view_matrix = p.computeViewMatrixFromYawPitchRoll(distance=self.cameraDistance,
															yaw=self.cameraYaw,
															pitch=self.cameraPitch,
															roll=self.cameraRoll,
															cameraTargetPosition=self.cameraTargetPosition,
															upAxisIndex=2)

	# returns an rgb image as a numpy array
	def get_image(self):
		
		_, _, rgba, _, _ = p.getCameraImage(width=self.cameraWidth,
												height=self.cameraHeight,
												viewMatrix=self.view_matrix,
												projectionMatrix=self.proj_matrix,
												renderer=p.ER_BULLET_HARDWARE_OPENGL,
												flags=p.ER_NO_SEGMENTATION_MASK)

		rgba = np.array(rgba, dtype=np.uint8).reshape((self.cameraWidth, self.cameraHeight, 4))
		rgb = rgba[:, :, :3]
		return rgb


# class for an onboard camera mounted to the robot's end-effector
# this camera moves with the robot during the simulation
class OnboardCamera():

	# the cameraDistance is the distance from the camera to its focal point
	# the cameraOffsetPosition is the displacement between the end-effector and the camera
	# the cameraOffsetQuaternion is the rotation between the end-effector and the camera
	def __init__(self, cameraDistance=0.2, cameraOffsetPosition=[0.05, 0.0, 0.0], cameraOffsetQuaternion=p.getQuaternionFromEuler([0, -np.pi/2, 0]), cameraWidth=256, cameraHeight=256):

		self.cameraDistance = cameraDistance
		self.cameraWidth = cameraWidth
		self.cameraHeight = cameraHeight
		self.cameraOffsetPosition = cameraOffsetPosition
		self.cameraOffsetQuaternion = cameraOffsetQuaternion

		self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
														aspect=self.cameraWidth / self.cameraHeight,
														nearVal=0.01,
														farVal=2.0)

	# returns an rgb image as a numpy array
	# input the position and orientation of the mounting point on the robot
	def get_image(self, ee_position, ee_quaternion):

		# update the camera position and orientation as the robot moves
		cam_pos, cam_orn = p.multiplyTransforms(ee_position,
												ee_quaternion,
												self.cameraOffsetPosition,
												self.cameraOffsetQuaternion)
		rot_mat = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
		cam_forward = rot_mat @ np.array([1, 0, 0])
		cam_up = rot_mat @ np.array([0, 0, 1])

		view_matrix = p.computeViewMatrix(cameraEyePosition=cam_pos,
											cameraTargetPosition=cam_pos + self.cameraDistance * cam_forward,
											cameraUpVector=cam_up)

		_, _, rgba, _, _ = p.getCameraImage(width=self.cameraWidth,
												height=self.cameraHeight,
												viewMatrix=view_matrix,
												projectionMatrix=self.proj_matrix,
												renderer=p.ER_BULLET_HARDWARE_OPENGL,
												flags=p.ER_NO_SEGMENTATION_MASK)

		rgba = np.array(rgba, dtype=np.uint8).reshape((self.cameraWidth, self.cameraHeight, 4))
		rgb = rgba[:, :, :3]
		return rgb
