"""
Open one cabinet door by id
"""

import traceback

from utils.robot_utils.door_skills import (approach_door, detect_handle, load_door_target,
                                           open_door_sequence)
from utils.robot_utils.container_skills import load_scene
from utils.robot_utils.robot_session import robot_session


def execute_search(door_id: int) -> bool:
    """
    Open a cabinet door
    """
    
    try:
        with robot_session() as robot:
            # Switch to position mode before executing door motions.
            robot.switch_to_position_mode()

            # ---1. Check which door to open and load the scene---
            print(f"Opening door {door_id}.")
            target = load_door_target(door_id, *load_scene())
            print(f"Furniture: {target.furniture_name}, centre {target.furniture_center}, "
                  f"door centre {target.center}")

            # ---2. Approach the door and detect the handle---
            front_normal = approach_door(robot, target)
            handle_pose, door_type = detect_handle(robot, target, front_normal)

            # ---3. Step to the swing side, grab the knob and pull it open---
            return open_door_sequence(robot, target, handle_pose, door_type)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # docker exec -it heuristic_khorana bash -c "source /opt/ros/humble/setup.bash && source /home/ws/install/setup.bash && ros2 run sam3_inference sam3_service.py"
    import sys
    execute_search(int(sys.argv[1]))
