import os
import time
import copy
import datetime
import numpy as np
from termcolor import colored

from .pi0 import PI0
from .utils import ObservationBuffer
from .cameras_subscriber import CamerasSubscriber
from src.franka_panda import Franka
from src.cameras import VideoRecorder


def deploy(robot: Franka, conn_robot, conn_gripper, cameras_sub, policy, task) -> None:
    p_task = copy.copy(task).replace(" ", "_").replace(".", "")
    print(colored(f"[Policy] ", "green") + f"Received task: {task}")
    robot.go2position(conn_robot)
    robot.send2gripper(conn_gripper, "o")
    gripper_state = 1
    os.makedirs(f"./baselines/pi0/videos/{p_task}", exist_ok=True)
    print(colored("[Policy] ", "green") + f"Save deploy to: ./baselines/pi0/videos/{p_task}")
    recorder = VideoRecorder(path=f"./baselines/pi0/videos/{p_task}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4", fps=20)

    # get current robot position
    state = robot.listen2robot(conn_robot)
    assert state is not None, "Failed to get robot state"
    start_position = state["q"].tolist() + [gripper_state]
    assert len(start_position) == 8, f"Expected 8 values for joint positions and gripper state, got {len(start_position)}"
    print(f"Current robot state: {start_position}")

    obs = {}
    frames = cameras_sub.get_last_obs()
    obs["image"] = np.array(frames["left_realsense_img_rgb"][-1])
    obs["image_gripper"] = np.array(frames["gripper_img_rgb"][-1])   
    obs['agent_pos'] = np.array(start_position)

    obs_buffer = ObservationBuffer(buffer_size=4, init_obs=obs)

    # main policy
    try:
        step_time = 1 / 20
        start_time = time.time()
        while True:
            try:
                curr_time = time.time()
                if curr_time - start_time >= step_time:
                    start_time = curr_time

                    obs = {}
                    frames = cameras_sub.get_last_obs()
                    obs["image"] = np.array(frames["left_realsense_img_rgb"][-1])
                    obs["image_gripper"] = np.array(frames["gripper_img_rgb"][-1])
                    state = robot.listen2robot(conn_robot)
                    recorder.add_frame(obs["image"])
                    if state is None:
                        print("Failed to get robot state, skipping this step")
                        continue
                    obs['agent_pos'] = np.array(state["q"].tolist() + [gripper_state])
                    print("Current robot state:", state["x"][2])
                    obs_buffer.add(obs)
                    obs = obs_buffer.get_buffer()

                    action = policy.get_action(obs)
                    print(f"Action from policy: {action}")
                    if action is None:
                        print("Policy done, exiting")
                        break

                    assert len(action) == 8
                    # if action[-1] >= 0.4: 
                    if action[-1] < -1.0 and gripper_state == 1:
                        robot.send2gripper(conn_gripper, "c")
                        gripper_state = -1
                    elif action[-1] > 0.8 and gripper_state == -1:
                        robot.send2gripper(conn_gripper, "o")
                        gripper_state = 1
                    robot.send2robot(conn_robot, action[:-1])
                    
            except KeyboardInterrupt:
                recorder.close()
                del recorder
                del obs_buffer
                time.sleep(1)
                robot.go2position(conn_robot)
                robot.send2gripper(conn_gripper, "o")
    finally:
        raise KeyboardInterrupt

def main() -> None:
    panda = Franka()
    print('[*] Connecting to robot...')
    conn_robot = panda.connect(8080)
    print('[*] Connecting to gripper...')
    conn_gripper = panda.connect(8081)

    # start cameras
    topics = ["gripper_img_rgb",
              "left_realsense_img_rgb"]
    cameras_sub = CamerasSubscriber(topics=topics,
                                   server_addr='localhost',
                                   port=8082)
    cameras_sub.start_thread()
    task = input("Enter the task:")
    policy = PI0(device="cuda", action_scale=14, name="pi0", prompt=task)
    while True:
        try:
            deploy(panda, conn_robot, conn_gripper, cameras_sub, policy, task)    
        except KeyboardInterrupt:
            con = input("Continue?(y/n)")
            if "y" in con.lower():
                continue
            else:
                break
    del policy

if __name__ == "__main__":
    main()