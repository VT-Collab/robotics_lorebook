import pybullet as p
import pybullet_data
import numpy as np
import os
import time
import yaml
import random
from ..cameras import ExternalCamera, VideoRecorder
from ..franka_panda import Panda
from ..objects import objects
import io
from contextlib import redirect_stdout
from dataclasses import dataclass, field, fields, _MISSING_TYPE


@dataclass
class PandaEnvConfig:
    baseStartPosition: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    baseStartOrientation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    control_dt: float = 1.0 / 240.0
    jointStartPositions: list[float] = field(
        default_factory=lambda: [
            0.0,
            0.0,
            0.0,
            -2 * np.pi / 4,
            0.0,
            np.pi / 2,
            np.pi / 4,
            0.0,
            0.0,
            0.04,
            0.04,
        ]
    )
    cameraDistance: float = 1.0
    cameraYaw: float = -40.0
    cameraPitch: float = -30.0
    cameraTargetPosition: list[float] = field(default_factory=lambda: [-0.5, 0.0, 0.2])
    ext_cameraDistance: float = 0.7
    ext_cameraYaw: float = -90.0
    ext_cameraPitch: float = -40.0
    ext_cameraTargetPosition: list[float] = field(
        default_factory=lambda: [0.7, 0.0, 0.2]
    )

    def __post_init__(self):
        # adapted from https://stackoverflow.com/questions/56665298
        for f in fields(self):
            if getattr(self, f.name) is None:
                if not isinstance(f.default, _MISSING_TYPE):
                    setattr(self, f.name, f.default)
                elif not isinstance(f.default_factory, _MISSING_TYPE):
                    setattr(self, f.name, f.default_factory())


class PandaEnv(object):
    def __init__(self, config: PandaEnvConfig = None, scene_config: str = "microwave.yml"):
        if config is None:
            config = PandaEnvConfig(
                ext_cameraDistance=1.5,
                ext_cameraYaw=90.0,
                ext_cameraPitch=-90.0,
                ext_cameraTargetPosition=[0.5, 0.0, 0.0],
            )
        self.config = config
        self._recorder = None
        self._init_pybullet(config)
        self._load_objects(config, scene_config)
        self.panda = Panda(
            basePosition=config.baseStartPosition,
            baseOrientation=self.p.getQuaternionFromEuler(config.baseStartOrientation),
            jointStartPositions=config.jointStartPositions,
        )
        self.external_camera = ExternalCamera(
            cameraDistance=config.ext_cameraDistance,
            cameraYaw=config.ext_cameraYaw,
            cameraPitch=config.ext_cameraPitch,
            cameraTargetPosition=config.ext_cameraTargetPosition,
        )
        # let the scene initialize
        for _ in range(100):
            self.p.stepSimulation()
            time.sleep(config.control_dt)

    def _init_pybullet(self, config: PandaEnvConfig) -> None:
        self.p = p
        self.physicsClient = self.p.connect(self.p.GUI)
        self.p.setGravity(0.0, 0.0, -9.81)
        self.p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.p.resetDebugVisualizerCamera(
            cameraDistance=config.cameraDistance,
            cameraYaw=config.cameraYaw,
            cameraPitch=config.cameraPitch,
            cameraTargetPosition=config.cameraTargetPosition,
        )

    def _load_objects(self, config: PandaEnvConfig, scene_config: str) -> None:
        def _sample_range(value):
            if isinstance(value, (list, tuple)) and len(value) == 2:
                min_vals, max_vals = value
                if isinstance(min_vals, (list, tuple)) and isinstance(max_vals, (list, tuple)):
                    if len(min_vals) != len(max_vals):
                        raise ValueError("Range bounds must have the same length")
                    return [
                        random.uniform(min(a, b), max(a, b))
                        for a, b in zip(min_vals, max_vals)
                    ]
            return value

        self.urdfRootPath = pybullet_data.getDataPath()
        plane = p.loadURDF(
            os.path.join(self.urdfRootPath, "plane.urdf"),
            basePosition=[0, 0, -0.625],
        )
        table = p.loadURDF(
            os.path.join(self.urdfRootPath, "table/table.urdf"),
            basePosition=[0.5, 0, -0.625],
        )
        config_path = f"config/scene/{scene_config}"
        raw_definitions = [
            ("plane", plane),
            ("table", table),
        ]
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            object_specs = data.get("objects", [])
            for spec in object_specs:
                name = spec.get("name")
                if name in {"plane", "table"}:
                    continue
                loader = str(spec.get("loader", "simple")).lower()
                urdf = spec.get("urdf")
                base_position = _sample_range(spec.get("basePosition", [0.0, 0.0, 0.0]))
                base_orientation_euler = _sample_range(
                    spec.get("baseOrientationEuler", None)
                )
                base_orientation = p.getQuaternionFromEuler(base_orientation_euler)
                global_scaling = spec.get("globalScaling", 0.08 if loader == "ycb" else 1.0)
                use_fixed_base = spec.get("useFixedBase", False)

                if loader == "simple":
                    obj = objects.SimpleObject(
                        f"{urdf}.urdf",
                        basePosition=base_position,
                        baseOrientation=base_orientation,
                        globalScaling=global_scaling,
                        useFixedBase=use_fixed_base,
                    )
                elif loader == "collab":
                    obj = objects.CollabObject(
                        f"{urdf}.urdf",
                        basePosition=base_position,
                        baseOrientation=base_orientation,
                        globalScaling=global_scaling,
                        useFixedBase=use_fixed_base,
                    )
                elif loader == "ycb":
                    obj = objects.YCBObject(
                        f"{urdf}.urdf",
                        basePosition=base_position,
                        baseOrientation=base_orientation,
                        globalScaling=global_scaling,
                        useFixedBase=use_fixed_base,
                    )
                elif loader == "robocasa":
                    obj = objects.RoboCasaObject,(
                        f"{urdf}.urdf",
                        basePosition=base_position,
                        baseOrientation=base_orientation,
                        globalScaling=global_scaling,
                        useFixedBase=use_fixed_base,
                    )
                else:
                    raise ValueError(f"Unsupported loader type: {loader}")

                raw_definitions.append((name, obj))
        else:
            raise FileNotFoundError(f"Scene config file not found: {config_path}")

        self.objects = []
        for label, obj in raw_definitions:
            if isinstance(obj, int):
                body_id = obj
            else:
                body_id = getattr(
                    obj, "id", getattr(obj, "uid", getattr(obj, "body_id", None))
                )
                if body_id is None:
                    for attr_name in dir(obj):
                        if not attr_name.startswith("__"):
                            val = getattr(obj, attr_name)
                            if isinstance(val, int) and val >= 0:
                                body_id = val
                                break
            if body_id is not None:
                self.objects.append({"id": body_id, "type": label, "ref": obj})
            else:
                print(f"Warning: Failed to find PyBullet ID for {label}")

    def step(self):
        self.p.stepSimulation()
        if self._recorder is not None:
            self._step_count += 1
            if self._step_count % self._record_every == 0:
                frame = self.get_image()
                self._recorder.add_frame(frame)

    def set_recorder(self, video_path: str = None, fps: int = 20):
        if video_path is None:
            if self._recorder is not None:  # Stop recording
                self._recorder.close()
                self._recorder = None
            return

        if self._recorder is not None:  # Start recording
            raise ValueError(
                "Recorder already exists. Please close it before setting a new one."
            )
        self._recorder = VideoRecorder(video_path, fps)
        sim_hz = 1.0 / self.config.control_dt
        self._record_every = max(1, int(round(sim_hz / fps)))
        self._step_count = 0

    def get_state(self) -> dict:
        return self.panda.get_state()

    def get_print_state(self) -> tuple:
        state = self.get_state()
        pos = state["ee-position"]
        orn = state["ee-euler"]
        return (pos, orn)

    def move_to_pose(self, ee_position: list[float], ee_euler: list[float]) -> tuple:
        position_threshold = 0.005
        orientation_threshold = 0.03
        target_pos = np.array(ee_position, dtype=float)
        target_euler = np.array(ee_euler, dtype=float)
        for _ in range(1000):
            state = self.get_state()
            current_pos = np.array(state["ee-position"], dtype=float)
            current_euler = np.array(state["ee-euler"], dtype=float)
            if (
                np.linalg.norm(current_pos - target_pos) <= position_threshold
                and np.linalg.norm(current_euler - target_euler) <= orientation_threshold
            ):
                return self.get_print_state()
            self.panda.move_to_pose(
                ee_position=ee_position,
                ee_quaternion=p.getQuaternionFromEuler(ee_euler),
                positionGain=0.01,
            )
            self.step()
            time.sleep(self.config.control_dt)
        return self.get_print_state()

    def side_align_vertical(self, angle: float) -> tuple:
        ee_euler = [np.pi / 2, 0.0, np.pi / 2 + angle]
        state = self.get_state()
        self.move_to_pose(state["ee-position"], ee_euler)
        return self.get_print_state()

    def side_align_horizontal(self, angle: float) -> tuple:
        ee_euler = [np.pi, -np.pi / 2, angle]
        state = self.get_state()
        self.move_to_pose(state["ee-position"], ee_euler)
        return self.get_print_state()

    def top_grasp(self, angle: float) -> tuple:
        ee_euler = [np.pi, 0.0, angle]
        state = self.get_state()
        self.move_to_pose(state["ee-position"], ee_euler)
        return self.get_print_state()

    def spin_gripper_inplace(self, theta: float) -> tuple:
        state = self.get_state()
        new_theta = theta + state["joint-position"][6]
        for _ in range(500):
            self.p.setJointMotorControl2(
                self.panda.panda,
                6,
                self.p.POSITION_CONTROL,
                targetPosition=new_theta,
                positionGain=0.01,
            )
            self.step()
            time.sleep(self.config.control_dt)
        return self.get_print_state()

    def spin_gripper(self, theta: float) -> tuple:
        """Rotates the end effector about the world Z axis by theta radians."""
        state = self.get_state()
        current_pos = state["ee-position"]
        current_orn_quat = state["ee-quaternion"]

        # 1. Define the rotation around the World Z axis
        # Note: If theta is absolute, remove the 'delta' logic.
        # Here we treat it as an offset to the current orientation.
        rot_offset = self.p.getQuaternionFromEuler([0, 0, theta])

        # 2. Multiply quaternions to apply the rotation in the world frame
        # New_Orientation = Rotation_Offset * Current_Orientation
        new_orn_quat = self.p.multiplyTransforms(
            [0, 0, 0], rot_offset, [0, 0, 0], current_orn_quat
        )[1]

        # 3. Use move_to_pose to reach the new orientation via IK
        # We pass the quaternion directly to ensure precision
        for _ in range(500):
            self.panda.move_to_pose(
                ee_position=current_pos,
                ee_quaternion=new_orn_quat,
                positionGain=0.01,
            )
            self.step()
            time.sleep(self.config.control_dt)

        return self.get_print_state()

    def move_to_position(self, position: list[float]) -> tuple:
        state = self.get_state()
        self.move_to_pose(position, state["ee-euler"])
        return self.get_print_state()

    def open_gripper(self) -> tuple:
        for _ in range(600):
            self.panda.open_gripper()
            self.step()
            time.sleep(self.config.control_dt)
        return self.get_print_state()

    def close_gripper(self) -> tuple:
        for _ in range(600):
            self.panda.close_gripper()
            self.step()
            time.sleep(self.config.control_dt)
        return self.get_print_state()

    def get_image(self) -> np.ndarray:
        return self.external_camera.get_image()

    def task_completed(self) -> None:
        print("Task Completed")
        return

    def get_checkpoint(self):
        return self.p.saveState()

    def restore_checkpoint(self, state_id):
        self.p.restoreState(stateId=state_id)
        self.reset_motors()

    def run_code(self, code: str) -> str:
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                exec(code)
        except KeyboardInterrupt:
            self.reset_motors()
            raise
        return buffer.getvalue()

    def reset_motors(self):
        """Overrides all motor targets with current positions to stop movement."""
        state = self.get_state()
        for i in range(7):
            self.p.setJointMotorControl2(
                self.panda.panda,
                i,
                self.p.POSITION_CONTROL,
                targetPosition=state["joint-position"][i],
                force=50,
            )
        for i in [9, 10]:
            self.p.setJointMotorControl2(
                self.panda.panda,
                i,
                self.p.POSITION_CONTROL,
                targetPosition=state["gripper"][0] if i == 9 else state["gripper"][1],
                force=20,
            )
