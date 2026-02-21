import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
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
robot.send2gripper(conn_gripper, 'c')
kp = 0.3

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
target_marker_id = None

# calibrate with 20 points
i = 0
while i < 20:
    print(colored("[Calibration] ", "green") + f"Capturing point {i+1}/20...")
    # move the robot to a random cartisian posution
    pos = np.random.uniform(low=[0.40, -0.3, 0.10], high=[0.55, 0.3, 0.43])
    orien = np.random.uniform(low=[-np.pi - np.pi / 4, -np.pi/5, -np.pi/4], high=[- np.pi + np.pi / 4, np.pi/6, np.pi/4])
    state = robot.readState(conn_robot)
    # pos = state["x"][:3] + np.array([0, 0.15, 0])
    # orien = state["angle"] + np.array([0, 0, 0])
    print(f"target_pos: {pos}")
    print(f"taget_orien: {orien}")
    for _ in range(500):
        q_curr_inv = Rotation.from_euler('xyz', state["x"][3:], degrees=False).inv()
        q_goal = Rotation.from_euler('xyz', orien, degrees=False)
        q_diff = q_goal * q_curr_inv
        delta_rot = q_diff.as_euler('xyz', degrees=False)
        xdot = np.concatenate([kp * (pos - state["x"][:3]), kp * delta_rot])
        if np.linalg.norm(xdot) < 0.005:
            break
        qdot = robot.xdot2qdot(xdot, state)
        robot.send2robot(conn_robot, qdot)
        state = robot.readState(conn_robot)

    time.sleep(2)  
    state = robot.readState(conn_robot)
    cartesian_pos = state["x"]
    print(colored("[Calibration] ", "green") + f"Robot position {i+1}/20: {cartesian_pos}")

    # Get the Aruco marker position in camera frame
    marker_poses_camera, _ = camera.read_ArUco()
    if marker_poses_camera is None or len(marker_poses_camera) == 0:
        print(colored("[Calibration] ", "yellow") + "No marker detected, skipping this sample.")
        continue

    if target_marker_id is None:
        if "id" in marker_poses_camera[0]:
            target_marker_id = int(marker_poses_camera[0]["id"])
            print(colored("[Calibration] ", "green") + f"Locking marker id = {target_marker_id}")
        else:
            target_marker_id = -1

    maker_pose_camera = None
    for pose in marker_poses_camera:
        if target_marker_id == -1:
            maker_pose_camera = marker_poses_camera[0]
            break
        if int(pose.get("id", -99999)) == target_marker_id:
            maker_pose_camera = pose
            break
    if maker_pose_camera is None:
        print(colored("[Calibration] ", "yellow") + f"Marker id {target_marker_id} not found, skipping sample.")
        continue

    # robot pose
    r_xyz, R_rm = robot.joint2pose(state["q"])
    T_rm = np.eye(4, dtype=np.float64)
    T_rm[:3, :3] = R_rm
    T_rm[:3, 3] = np.array(r_xyz, dtype=np.float64)

    # camera marker pose
    cx, cy, cz = maker_pose_camera["x"], maker_pose_camera["y"], maker_pose_camera["z"]
    R_cm = np.asarray(maker_pose_camera["R"], dtype=np.float64).reshape(3, 3)
    T_cm = np.eye(4, dtype=np.float64)
    T_cm[:3, :3] = R_cm
    T_cm[:3, 3] = np.array([cx, cy, cz], dtype=np.float64)

    R_rm_list.append(R_rm)
    R_cm_list.append(R_cm)
    t_rm_list.append(np.array(r_xyz, dtype=np.float64))
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

trans_err = []
rot_err_deg = []
for R_rm, R_cm, t_rm, t_cm in zip(R_rm_list, R_cm_list, t_rm_list, t_cm_list):
    T_rm = np.eye(4, dtype=np.float64)
    T_rm[:3, :3] = R_rm
    T_rm[:3, 3] = t_rm

    T_cm = np.eye(4, dtype=np.float64)
    T_cm[:3, :3] = R_cm
    T_cm[:3, 3] = t_cm

    T_pred = C2R @ T_cm
    dT = np.linalg.inv(T_pred) @ T_rm

    dt = float(np.linalg.norm(dT[:3, 3]))
    cosang = np.clip((np.trace(dT[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    dang = float(np.degrees(np.arccos(cosang)))
    trans_err.append(dt)
    rot_err_deg.append(dang)

for k, (dt, dang) in enumerate(zip(trans_err, rot_err_deg), start=1):
    print(colored("[Calibration] ", "cyan") + f"Sample {k:02d}: trans_err={dt:.6f} m, rot_err={dang:.3f} deg")

np.save("C2R.npy", C2R)
print(colored("[Calibration] ", "green") + f"Calculated C2R transform matrix:\n{C2R}")
print(colored("[Calibration] ", "green") + f"Saved C2R.npy using {len(R_rm_list)} samples")
print(colored("[Calibration] ", "green") + f"Mean residual: trans={np.mean(trans_err):.6f} m, rot={np.mean(rot_err_deg):.3f} deg")
print(C2R)


    


    