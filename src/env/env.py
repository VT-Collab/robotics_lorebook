import pybullet as p
import pybullet_data
import numpy as np
import os
import time
from ..cameras import ExternalCamera
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
    def __init__(self, config: PandaEnvConfig = None):
        if config is None:
            config = PandaEnvConfig()
        self.config = config
        self._init_pybullet(config)
        self._load_objects(config)
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

    def _load_objects(self, config: PandaEnvConfig) -> None:
        self.urdfRootPath = pybullet_data.getDataPath()
        plane = p.loadURDF(
            os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.625]
        )
        table = p.loadURDF(
            os.path.join(self.urdfRootPath, "table/table.urdf"),
            basePosition=[0.5, 0, -0.625],
        )
        cube_wrapper = objects.SimpleObject(
            "cube.urdf",
            basePosition=[0.5, -0.3, 0.025],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0.7]),
        )
        cabinet_wrapper = objects.CollabObject(
            "cabinet.urdf",
            basePosition=[0.9, 0.0, 0.2],
            baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]),
        )

        raw_definitions = [
            ("plane", plane),
            ("table", table),
            ("cube", cube_wrapper),
            ("cabinet", cabinet_wrapper),
        ]

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

    def get_state(self) -> dict:
        return self.panda.get_state()

    def get_print_state(self) -> tuple:
        state = self.get_state()
        pos = state["ee-position"]
        orn = state["ee-euler"]
        return (pos, orn)

    def move_to_pose(self, ee_position: list[float], ee_euler: list[float]) -> tuple:
        for _ in range(1000):
            self.panda.move_to_pose(
                ee_position=ee_position,
                ee_quaternion=p.getQuaternionFromEuler(ee_euler),
                positionGain=0.01,
            )
            p.stepSimulation()
            time.sleep(self.config.control_dt)
        return self.get_print_state()

    def side_grasp_vertical(self, angle: float) -> tuple:
        ee_euler = [np.pi / 2, 0.0, np.pi / 2 + angle]
        state = self.get_state()
        self.move_to_pose(state["ee-position"], ee_euler)
        return self.get_print_state()

    def side_grasp_horizontal(self, angle: float) -> tuple:
        ee_euler = [np.pi, -np.pi / 2, angle]
        state = self.get_state()
        self.move_to_pose(state["ee-position"], ee_euler)
        return self.get_print_state()

    def top_grasp(self, angle: float) -> tuple:
        ee_euler = [np.pi, 0.0, angle]
        state = self.get_state()
        self.move_to_pose(state["ee-position"], ee_euler)
        return self.get_print_state()

    def spin_gripper(self, theta: float) -> tuple:
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
            self.p.stepSimulation()
            time.sleep(self.config.control_dt)
        return self.get_print_state()

    def move_to_position(self, position: list[float]) -> tuple:
        state = self.get_state()
        self.move_to_pose(position, state["ee-euler"])
        return self.get_print_state()

    def open_gripper(self) -> tuple:
        for _ in range(300):
            self.panda.open_gripper()
            self.p.stepSimulation()
            time.sleep(self.config.control_dt)
        return self.get_print_state()

    def close_gripper(self) -> tuple:
        for _ in range(300):
            self.panda.close_gripper()
            self.p.stepSimulation()
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

    def run_code(self, code: str) -> str:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exec(code)
        return buffer.getvalue()
