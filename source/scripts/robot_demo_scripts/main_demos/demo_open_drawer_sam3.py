import traceback
from utils.robot_utils.drawer_skills import (approach_drawer, center_on_handle,
                                                    close_drawer, detect_handle,
                                                    grasp_from_drawer, load_drawer_target,
                                                    load_scene, measure_handle)
from utils.robot_utils.robot_session import robot_session


OBJECT = 'tennis ball'


def execute_search(drawer_id: int) -> bool:
    """
    Open a drawer and grasp the object inside it.

    Args:
        drawer_id (int): id of the drawer in the scene graph

    Returns:
        bool: True if the object ended up held, False otherwise
    """

    try:
        with robot_session() as robot:
            # Switch to position mode before executing drawer motions.
            robot.switch_to_position_mode()

            # ---1. Check which drawer to open and load the scene---
            print(f"Opening drawer {drawer_id}.")
            target = load_drawer_target(drawer_id, *load_scene())

            # --2. Approach the drawer and detect the handle---
            front_normal = approach_drawer(robot, target)
            handle_pose, drawer_type = detect_handle(robot, target, front_normal)

            # ---3. Center the gripper on the handle---
            handle_pose = center_on_handle(robot, target, handle_pose)

            # ---4. Grasp the object inside the drawer---
            grasped = grasp_from_drawer(robot, handle_pose, OBJECT)

            # ---5. Close the drawer---
            close_drawer(robot, handle_pose)
            return grasped

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # docker run -p 5004:5004 --gpus all -it craiden/yolodrawer:v1.0 python3 app.py
    import sys
    execute_search(int(sys.argv[1]))
