from src.env.real_env import RealEnv
import numpy as np

def test_real_env():
    # Initialize the RealEnv object
    env = RealEnv()

    # Test get_state and get_print_state
    print("Testing get_state and get_print_state")
    state = env.get_state()
    print("State:", state)
    print("Print State:", env.get_print_state())

    # Test move_to_pose
    print("Testing move_to_pose")
    ee_position = [0.75, 0.12, 0.2]
    print("Move to pose result:", env.move_to_pose(ee_position, state["x"][3:]))

    # # Test side_align_vertical
    # print("Testing side_align_vertical")
    # print("Side align vertical result:", env.side_align_vertical(np.pi/2 + np.pi/6))

    # Test side_align_horizontal
    print("Testing side_align_horizontal")
    print("Side align horizontal result:", env.side_align_horizontal(np.pi / 3))

    # # Test top_grasp
    # print("Testing top_grasp")
    # print("Top grasp result:", env.top_grasp(np.pi / 6))

    # # Test spin_gripper_inplace
    # print("Testing spin_gripper_inplace")
    # print("Spin gripper inplace result:", env.spin_gripper_inplace(np.pi / 2))

    # # Test spin_gripper
    # print("Testing spin_gripper")
    # print("Spin gripper result:", env.spin_gripper(np.pi / 2))

    # # Test move_to_position
    # print("Testing move_to_position")
    # position = [0.3, 0.3, 0.3]
    # print("Move to position result:", env.move_to_position(position))

    # # Test open_gripper
    # print("Testing open_gripper")
    # print("Open gripper result:", env.open_gripper())

    # # Test close_gripper
    # print("Testing close_gripper")
    # print("Close gripper result:", env.close_gripper())

if __name__ == "__main__":
    test_real_env()