import io
import time
import numpy as np
from typing import Optional
from termcolor import colored
from threading import Thread, Event, Lock
from contextlib import redirect_stdout, contextmanager
from ..cameras import OrbbecCamera, VideoRecorder
from ..franka_panda import Franka
from scipy.spatial.transform import Rotation
from dataclasses import dataclass, field, fields, _MISSING_TYPE


class RealEnv(object):
    def __init__(self, config: None = None, scene_config: Optional[str] = None):
        self.panda = Franka()
        print('[*] Connecting to robot...')
        self.conn_robot = self.panda.connect(8080)
        print('[*] Connecting to gripper...')
        self.conn_gripper = self.panda.connect(8081)
        self.panda.go2position(self.conn_robot)
        self.panda.send2gripper(self.conn_gripper, 'o')
        self.gripper_open = float(1)
        self.objects = {}
        self.ckpt_dict = {}
        self.previous_marked_pos = None
        self._recorder = None
        self._record_thread = None
        self._record_stop_event = Event()
        self._record_lock = Lock()
        self._record_motion_depth = 0

        self.camera = OrbbecCamera(
            output_fps = 20, 
            max_frames = 20 * 60,
        )
        self.camera.start()
        time.sleep(8)

    def _extract_camera_frame(self, payload) -> np.ndarray | None:
        if payload is None:
            return None
        if isinstance(payload, dict):
            payload = payload.get("frame")
        if payload is None:
            return None
        frame = np.asarray(payload)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            return None
        return frame

    @contextmanager
    def _record_motion_scope(self):
        with self._record_lock:
            self._record_motion_depth += 1
        try:
            yield
        finally:
            with self._record_lock:
                self._record_motion_depth = max(0, self._record_motion_depth - 1)

    def _recorder_loop(self, fps: int = 20) -> None:
        interval = 1.0 / max(1, fps)
        while not self._record_stop_event.is_set():
            frame = self._extract_camera_frame(self.camera.get_image())
            with self._record_lock:
                should_record = self._recorder is not None and self._record_motion_depth > 0
                recorder = self._recorder
            if should_record and recorder is not None:
                if frame is not None:
                    recorder.add_frame(frame)
            time.sleep(interval)

    def set_recorder(self, video_path: str | None = None, fps: int = 20) -> None:
        with self._record_lock:
            is_running = self._recorder is not None

        if is_running:
            self._record_stop_event.set()
            if self._record_thread is not None:
                self._record_thread.join(timeout=2.0)
            with self._record_lock:
                recorder = self._recorder
                self._recorder = None
                self._record_motion_depth = 0
            if recorder is not None:
                recorder.close()
            self._record_thread = None
            print(colored("[RealEnv] ", "yellow") + f"Stopped recording.")
            return

        if video_path is None:
            video_path = f"real_env_{int(time.time())}.mp4"

        self._record_fps = max(1, int(fps))
        with self._record_lock:
            self._recorder = VideoRecorder(video_path, self._record_fps)
            self._record_motion_depth = 0
            print(colored("[RealEnv] ", "green") + f"Started recording to {video_path} at {self._record_fps} FPS.")
        self._record_stop_event.clear()
        self._record_thread = Thread(target=self._recorder_loop, args=(fps,), daemon=True)
        self._record_thread.start()

    def _load_ArUco(self) -> None:
        use_previous = input("Record new ArUco marker positions? (y/n): ")
        if "n" in use_previous.lower():
            return
        while True:
            label = input("Enter a label for the ArUco marker positions (e.g., 'microwave handle'): ")
            print("Waiting for 10 secs to place ArUco markers", end="", flush=True)
            for _ in range(10):
                time.sleep(1)
                print(".", end="", flush=True)
            print()
            marker_poses_camera, marker_poses_robot = self.camera.read_ArUco()
            if len(marker_poses_robot) == 0:
                use_previous = input("Could not detect marker. Record new ArUco marker positions? (y/n): ")
                if "n" in use_previous.lower():
                    marker_poses_robot = self.previous_marked_pos[1]
                    break
            else:
                break
        self.previous_marked_pos = (label, marker_poses_robot)
        return 

    def load_objects(self) -> None:
        # get objects from the orbbec camera and save them to self.objects
        response = self.camera.find_objects()
        objects = response["objects"]
        bbox_dict = response["bbox"]
        self.objects = []
        for name, pos in objects.items ():
            bbox = bbox_dict.get(name, None)
            obj = {"type": name, "pos": pos, "bbox": bbox}
            self.objects.append(obj)
        self._load_ArUco()

    def get_state(self) -> dict:
        state = self.panda.readState(self.conn_robot)
        state["gripper"] = [self.gripper_open]
        return state

    def get_print_state(self) -> tuple:
        state = self.get_state()
        pos = state["x"][:3]
        orn = state["x"][3:]
        return (pos, orn)

    def move_to_pose(self, ee_position: list[float], ee_euler: list[float]) -> tuple:
        position_threshold = 0.01
        orientation_threshold = 0.2
        linear_speed = 0.03
        angular_kp = 0.3
        target_pos = np.array(ee_position, dtype=float)
        target_euler = np.array(ee_euler, dtype=float)
        for _ in range(1000):
            state = self.get_state()
            q_curr_inv = Rotation.from_euler('xyz', state["x"][3:], degrees=False).inv()
            q_goal = Rotation.from_euler('xyz', target_euler, degrees=False)
            q_diff = q_goal * q_curr_inv
            delta_rot = q_diff.as_euler('xyz', degrees=False)
            delta_pos = target_pos - state["x"][:3]
            pos_norm = np.linalg.norm(delta_pos)
            if (
                    pos_norm <= position_threshold
                    and np.linalg.norm(target_euler - state["x"][3:]) <= orientation_threshold
                ):
                    return self.get_print_state()
            if pos_norm > position_threshold:
                linear_cmd = linear_speed * (delta_pos / pos_norm)
            else:
                linear_cmd = np.zeros(3)
            angular_cmd = angular_kp * delta_rot
            xdot = np.concatenate([linear_cmd, angular_cmd])
            qdot = self.panda.xdot2qdot(xdot, state)
            self.panda.send2robot(self.conn_robot, qdot)
        return self.get_print_state()
    
    def _move_to_pose_restricted(self, ee_position: list[float], ee_euler: list[float]) -> tuple:
        position_threshold = 0.01
        orientation_threshold = 0.2
        linear_speed = 0.03
        angular_kp = 0.3
        target_pos = np.array(ee_position, dtype=float)
        target_euler = np.array(ee_euler, dtype=float)
        for _ in range(1000):
            state = self.get_state()
            q_curr_inv = Rotation.from_euler('xyz', state["x"][3:], degrees=False).inv()
            q_goal = Rotation.from_euler('xyz', target_euler, degrees=False)
            q_diff = q_goal * q_curr_inv
            delta_rot = q_diff.as_euler('xyz', degrees=False)
            delta_pos = target_pos - state["x"][:3]
            pos_norm = np.linalg.norm(delta_pos)
            if (
                    pos_norm <= position_threshold
                    and np.linalg.norm(target_euler - state["x"][3:]) <= orientation_threshold
                ):
                    return self.get_print_state()
            if pos_norm > position_threshold:
                linear_cmd = linear_speed * (delta_pos / pos_norm)
            else:
                linear_cmd = np.zeros(3)
            angular_cmd = angular_kp * delta_rot
            xdot = np.concatenate([linear_cmd, angular_cmd])
            qdot = self.panda.xdot2qdot(xdot, state)
            if state["q"][-4] <= -2.8 and qdot[-4] < 0:
                qdot[-4] = 0.0
            if state["q"][-4] >= -0.11 and qdot[-4] > 0:
                qdot[-4] = 0.0
            if state["q"][-3] >= 2.8 and qdot[-3] > 0:
                qdot[-3] = 0.0
            if state["q"][-2] >= 3.6 and qdot[-2] > 0:
                qdot[-2] = 0.0 
            if state["q"][-1] >= 2.8 and qdot[-1] > 0:
                qdot[-1] = 0.0
            if state["q"][1] >= 1.6 and qdot[1] > 0:
                qdot[1] = 0.0
            if state["q"][0] <= -2.8 and qdot[0] < 0:
                qdot[0] = 0.0
            self.panda.send2robot(self.conn_robot, qdot)
        return self.get_print_state()
    
    def goto_anchor_state(self):
        anchor_state = np.array([-1.39009, 0.681142, 1.38247, -1.50721, -0.676106, 1.66416,0.784704])
        NUM_JOINTS = 7
        for i in range(NUM_JOINTS - 2, 0, -1):
            curr_state = self.get_state()["q"]
            goal_state = np.concatenate((curr_state[:i], anchor_state[i:]))
            print()
            self.panda.go2position(self.conn_robot, goal_state)
    
    # def side_align_vertical(self, angle: float) -> tuple:
    #     # ee_euler = [np.pi - np.pi / 4, 0.0, np.pi / 2 + 0.2 * angle]
    #     ee_euler = [-np.pi / 2, 0.0, - np.pi / 2 + angle]
    #     state = self.get_state()
    #     # self.panda.go2position(self.conn_robot, np.array([-1.59057, 0.621653, 1.6322, -1.59415, -1.57361, 1.50474, 1.01073]))
    #     self.panda.go2position(self.conn_robot, np.array([-1.62409, 0.621326, 1.50764, -1.65576, 1.53118, 2.93527, 0.13083]))
    #     # self.goto_anchor_state()
    #     # self.move_to_pose(state["x"][:3], state["x"][3:])
    #     print(f"curr state: {self.get_state()['x'][3:]}")
    #     self._move_to_pose_restricted(state["x"][:3], ee_euler)
    #     return self.get_print_state()

    # def side_align_vertical(self, angle: float) -> tuple:
    #     angle = abs(angle) 
    #     assert angle >= 0, "Apply an angle in yaw that is larger than 0"
    #     state = self.get_state()
    #     target_pos = state["x"][:3]
    #     self.panda.go2position(self.conn_robot, np.array([-1.41348,0.0412583,0.876479,-2.23895,2.24392,2.27377,-0.0687361]))
    #     state = self.get_state()
    #     target_orien = state["x"][3:]
    #     target_orien[-1] = target_orien[-1] + angle
    #     self._move_to_pose_restricted(target_pos, target_orien)
    #     return self.get_print_state()

    def side_align_vertical(self, angle:float) -> tuple:
        state = self.get_state()
        target_pos = state["x"][:3]
        self.panda.go2position(self.conn_robot, np.array([0.113923, -0.303658, -0.17232, -2.4721, -0.0153278, 2.93583, 2.33438]))
        state = self.get_state()
        target_orien = state["x"][3:]
        target_orien[-1] = target_orien[-1] + angle
        self._move_to_pose_restricted(target_pos, target_orien)
        return self.get_print_state()
    
    # def side_align_horizontal(self, angle: float) -> tuple:
    #     angle = abs(angle)
    #     assert angle >= 0, "Apply an angle in yaw that is larger than 0"
    #     state = self.get_state()
    #     target_pos = state["x"][:3]
    #     self.panda.go2position(self.conn_robot, np.array([-1.41348,0.0412583,0.876479,-2.23895,2.24392,2.27377,-0.0687361 - np.pi / 2]))
    #     state = self.get_state()
    #     target_orien = state["x"][3:]
    #     target_orien[-1] = target_orien[-1] + angle
    #     self._move_to_pose_restricted(target_pos, target_orien)
    #     return self.get_print_state()

    def side_align_horizontal(self, angle:float) -> tuple:
        state = self.get_state()
        target_pos = state["x"][:3]
        self.panda.go2position(self.conn_robot, np.array([0.113923, -0.303658, -0.17232, -2.4721, -0.0153278, 2.93583, 2.33438 - np.pi/2]))
        state = self.get_state()
        target_orien = state["x"][3:]
        target_orien[-1] = target_orien[-1] + angle
        self._move_to_pose_restricted(target_pos, target_orien)
        return self.get_print_state()
    
    
    def top_grasp(self, angle: float) -> tuple:
        ee_euler = [np.pi, 0.0, angle]
        state = self.get_state()   
        self.panda.go2position(self.conn_robot, np.array([0.200227, 0.000754162, -0.314666, -1.67235, 0.062345, 1.71818, 0.720979]))
        self._move_to_pose_restricted(state["x"][:3], ee_euler)
        return self.get_print_state()
    
    def spin_gripper_inplace(self, theta: float) -> tuple:
        state = self.get_state()
        new_theta = theta + state["q"][6]
        for _ in range(1000):
            state = self.get_state()
            qdot = np.zeros(7)
            qdot[6] = 0.3 * (new_theta - state["q"][6])
            if np.linalg.norm(new_theta - state["q"][6]) <= 0.05:
                return self.get_print_state()
            self.panda.send2robot(self.conn_robot, qdot)
        return self.get_print_state()
    
    def spin_gripper(self, theta: float) -> tuple:  
        state = self.get_state()
        current_pos = state["x"][:3]
        current_orn_euler = state["x"][3:]
        current_orn_quat = Rotation.from_euler('xyz', current_orn_euler, degrees=False).as_quat()

        rot_offset = Rotation.from_euler('xyz', [0, 0, theta], degrees=False)
        new_orn_quat = (rot_offset * Rotation.from_quat(current_orn_quat)).as_quat()
        new_orn_euler = Rotation.from_quat(new_orn_quat).as_euler('xyz', degrees=False)
        self.move_to_pose(current_pos, new_orn_euler)
        return self.get_print_state()
    
    def move_to_position(self, position: list[float]) -> tuple:
        state = self.get_state()
        position[0] = position[0] + 0.02
        self._move_to_pose_restricted(position, state["x"][3:])
        return self.get_print_state()
    
    def open_gripper(self) -> tuple:
        self.panda.send2gripper(self.conn_gripper, 'o')
        self.gripper_open = float(1)
        time.sleep(2)
        return self.get_print_state()

    def close_gripper(self) -> tuple:
        self.panda.send2gripper(self.conn_gripper, 'c')
        self.gripper_open = float(0)
        time.sleep(2)
        return self.get_print_state()
    
    def get_image(self) -> np.ndarray:
        frame = self._extract_camera_frame(self.camera.get_image())
        if frame is None:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return frame
    
    def task_completed(self) -> None:
        print("Task Complete!")
        return
    
    def get_checkpoint(self) -> str:
        state = self.get_state()
        robot_state = state["q"].tolist() + state["gripper"]
        num_ckpts = len(self.ckpt_dict)
        ckpt_name = f"ckpt_{num_ckpts}"
        self.ckpt_dict[ckpt_name] = robot_state
        return ckpt_name

    def restore_checkpoint(self, ckpt_name: str) -> None:
        if ckpt_name not in self.ckpt_dict:
            print(f"Checkpoint {ckpt_name} does not exist.")
            return
        self.open_gripper()
        robot_state = self.ckpt_dict[ckpt_name]
        q = robot_state[:-1]
        gripper_open = robot_state[-1]
        if gripper_open >= 0.5:
            self.open_gripper()
        else:
            print("Waiting for 10 secs to put object back", end="", flush=True)
            for _ in range(10):
                time.sleep(1)
                print(".", end="", flush=True)
            print()
            self.close_gripper()
        for _ in range(1000):
            state = self.get_state()
            qdot = 0.3 * (np.array(q) - state["q"])
            if np.linalg.norm(np.array(q) - state["q"]) <= 0.05:
                break
            self.panda.send2robot(self.conn_robot, qdot)

    def remove_checkpoint(self, ckpt_name: str) -> None:
        del self.ckpt_dict[ckpt_name]
    
    def run_code(self, code: str) -> str:
        with self._record_motion_scope():
            buffer = io.StringIO()
            new_code = "import math\nimport numpy as np\n"+code
            try:
                with redirect_stdout(buffer):
                    exec(new_code)
            except Exception as e:
                if isinstance(Exception, KeyboardInterrupt):
                    pass
                else: 
                    print(e)
                    raise e
            except KeyboardInterrupt:
                raise
            return buffer.getvalue()




    
        

    
    
