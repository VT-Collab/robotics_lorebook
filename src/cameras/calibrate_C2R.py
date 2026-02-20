import time
import cv2
import numpy as np
from termcolor import colored

from ..franka_panda import Franka
from ..cameras import OrbbecCamera

# initialize the robot and camera
robot = Franka()
print('[*] Connecting to robot...')
conn_robot = robot.connect(8080)
print('[*] Connecting to gripper...')
conn_gripper = robot.connect(8081)
robot.go2position(conn_robot)
kp = 0.05

camera = OrbbecCamera(
    output_fps = 20, 
    max_frames = 20 * 60,
)
camera.start()
time.sleep(8)

R_rm_list = []
R_cm_list = []
t_rm_list = []
t_cm_list = []

# calibrate with 20 points
i = 0
while i < 20:
    print(colored("[Calibration] ", "green") + f"Capturing point {i+1}/20...")
    # move the robot to a random cartisian posution
    pos = np.random.uniform(low=[0.35, -0.3, 0.05], high=[0.7, 0.3, 0.5])
    orien = np.random.uniform(low=[-np.pi, -np.pi, -np.pi], high=[np.pi, np.pi, np.pi])
    state = robot.readState(conn_robot)
    for _ in range(100):
        xdot = np.concatenate([kp * (pos - state["x"][:3]), kp * (orien - state["x"][3:])])
        if np.linalg.norm(xdot) < 0.01:
            break
        qdot = robot.xdot2qdot(xdot, state)
        robot.send2robot(conn_robot, qdot)
        state = robot.readState(conn_robot)

    time.sleep(2)
    cartesian_pos = robot.readState(conn_robot)["x"]
    print(colored("[Calibration] ", "green") + f"Robot position {i+1}/20: {cartesian_pos}")

    # Get the Aruco marker position in camera frame
    marker_poses_camera, _ = camera.read_ArUco()
    if marker_poses_camera is None or len(marker_poses_camera) == 0:
        print(colored("[Calibration] ", "yellow") + "No marker detected, skipping this sample.")
        continue
    maker_pose_camera = marker_poses_camera[0]  # we only use the first detected marker

    # robot pose (x, y, z, euler xyz)
    rx, ry, rz = cartesian_pos[:3]
    alpha, beta, gamma = cartesian_pos[3:6]
    ca, cb, cg = np.cos([alpha, beta, gamma])
    sa, sb, sg = np.sin([alpha, beta, gamma])
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float64)
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=np.float64)
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]], dtype=np.float64)
    R_rm = Rz @ Ry @ Rx
    T_rm = np.eye(4, dtype=np.float64)
    T_rm[:3, :3] = R_rm
    T_rm[:3, 3] = np.array([rx, ry, rz], dtype=np.float64)

    # camera marker pose
    cx, cy, cz = maker_pose_camera["x"], maker_pose_camera["y"], maker_pose_camera["z"]
    R_cm = np.asarray(maker_pose_camera["R"], dtype=np.float64).reshape(3, 3)
    T_cm = np.eye(4, dtype=np.float64)
    T_cm[:3, :3] = R_cm
    T_cm[:3, 3] = np.array([cx, cy, cz], dtype=np.float64)

    R_rm_list.append(R_rm)
    R_cm_list.append(R_cm)
    t_rm_list.append(np.array([rx, ry, rz], dtype=np.float64))
    t_cm_list.append(np.array([cx, cy, cz], dtype=np.float64))
    i += 1


R_cross = np.zeros((3, 3), dtype=np.float64)
for R_rm, R_cm in zip(R_rm_list, R_cm_list):
    R_cross += R_rm @ R_cm.T
U, _, Vt = np.linalg.svd(R_cross)
R_avg = U @ Vt
if np.linalg.det(R_avg) < 0:
    U[:, -1] *= -1
    R_avg = U @ Vt

t_terms = []
for t_rm, t_cm in zip(t_rm_list, t_cm_list):
    t_terms.append(t_rm - R_avg @ t_cm)
t_avg = np.mean(t_terms, axis=0)

C2R = np.eye(4, dtype=np.float64)
C2R[:3, :3] = R_avg
C2R[:3, 3] = t_avg

np.save("C2R.npy", C2R)
print(colored("[Calibration] ", "green") + f"Calculated C2R transform matrix:\n{C2R}")
print(colored("[Calibration] ", "green") + f"Saved C2R.npy using {len(R_rm_list)} samples")
print(C2R)


    


    