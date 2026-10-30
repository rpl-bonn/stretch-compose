"""
The steps that are a door's own: stand off to the swing side, grab the knob, swing it open.

Everything up to the handle - loading the scene, driving to the container, detecting and
measuring the handle - is shared with drawers and lives in container_skills. What differs is
what happens next: a drawer comes straight out along the arm, while a door swings around a
hinge, so the robot has to reposition to the side the door opens towards before it pulls.

The usual order:

    target       = load_door_target(door_id, *load_scene())
    front_normal = approach_door(robot, target)
    handle_pose, door_type = detect_handle(robot, target, front_normal)
    open_door_sequence(robot, target, handle_pose, door_type)
"""

import time

import numpy as np
from utils.robot_utils.run_visual_servo import servo_grasp_object

from utils.coordinates import Pose3D, pose_distanced
from utils.robot_utils import searchnet_planning
from utils.robot_utils.advanced_movement import (look_for_door, move_in_front_of, move_in_side_of,
                                                 open_door, extension_to_handle, sam3_box_detector)
from utils.robot_utils.basic_movement import move_arm, set_gripper, spin_until_complete
from utils.robot_utils.basic_perception import correct_vertical_offset, correct_offsets_base_frame
from utils.robot_utils.container_skills import (GRAPH_DIR, ContainerTarget, approach_container,
                                                load_container_target, measure_handle,
                                                plan_body_pose)
# Re-exported so a door demo can pull its whole sequence from one module.
from utils.robot_utils.container_skills import load_scene  # noqa: F401
from utils.robot_utils.container_skills import detect_handle as detect_container_handle
from utils.robot_utils.robot_session import Robot
from utils.zero_shot_object_detection_sam3 import (detect_door_handle_sam3, get_position_from_head_detection,
                                                   sam3_detect_object, robust_depth_m, intrinsics_from_camera)
from utils.robot_utils.container_skills import capture_gripper_view

DoorTarget = ContainerTarget

PROMPTS = ["door", "drawer", "knob", "handle"]
KNOB_PROMPTS = ["knob", "handle", "circle", "wooden circle"]

APPROACH_DISTANCE = 0.6
HANDLE_DISTANCE = 0.7

GRIPPER_OPEN = 0.65
GRIPPER_CLOSED = -0.35


def load_door_target(door_id: int, graph_data: dict, scene_data: dict,
                     graph_dir: str = GRAPH_DIR) -> DoorTarget:
    """
    Resolve a door id to its centroid and the furniture it belongs to
    """

    return load_container_target(door_id, graph_data, scene_data, graph_dir)


def approach_door(robot: Robot, target: DoorTarget, dist: float = APPROACH_DISTANCE,
                  attempts: int = 5, tolerance: float = 0.5) -> np.ndarray:
    """
    Drive the robot in front of the door 
    """

    front_normal = approach_container(robot, target, searchnet_planning.plan_door_search, dist,
                                      roll=0.10, lift=0.09, attempts=attempts, tolerance=tolerance)
    look_for_door(robot.pose, wrist=0.01)
    return front_normal


def detect_handle(robot: Robot, target: DoorTarget, front_normal: np.ndarray,
                  prompts: list = PROMPTS) -> tuple[Pose3D, str]:

    return detect_container_handle(robot, target, front_normal, detect_door_handle_sam3, prompts)


def body_pose_beside_door(body_pose: Pose3D, handle_pose: Pose3D, door_type: str) -> Pose3D:
    """
    Where to stand to swing the door
    """

    side = 1.0 if door_type == "left" else -1.0
    x1, y1, z1 = body_pose.coordinates
    x2, y2, _ = handle_pose.coordinates
    dx, dy = x2 - x1, y2 - y1
    print(f"DIFFERENCE IN dx: {dx}, dy: {dy}")

    if abs(dx) < abs(dy):
        new_x = x1 + side * (dy / 2)
        new_y = y1 + dy / 2
    else:
        new_x = x1 + dx / 2
        new_y = y1 - side * (dx / 2)

    new_pose = Pose3D(np.array([new_x, new_y, z1]))
    new_pose.set_rot_from_direction(body_pose.direction())
    print(f"NEW POSE ({door_type} handle): {new_pose}")
    return new_pose


def grab_knob(robot: Robot, handle_pose: Pose3D, prompts: list = KNOB_PROMPTS,
              roll: float = 0.0) -> None:
    """
    Reach the knob, re-measure it up close, line the gripper up and close on it.
    """
    
    move_arm(robot.position, handle_pose, roll=roll)
    print("Arm moved to handle pose for opening door.")
    time.sleep(2.0)
    set_gripper(robot.pose, GRIPPER_OPEN)
    time.sleep(2.0)

    correct_offsets_base_frame(robot.tf, robot.pose, measure_handle(robot, prompts=prompts))
    correct_vertical_offset(robot.tf, robot.pose, measure_handle(robot, prompts=prompts))

    time.sleep(2.0)
    grasped = servo_grasp_object(pose_node=robot.pose, switch_mode=False, object_name="door knob")
    print(f"Visual servo grasp {'succeeded' if grasped else 'failed'}.")
    set_gripper(robot.pose, 0.0)
    time.sleep(1.0)
    set_gripper(robot.pose, GRIPPER_CLOSED)
    time.sleep(2.0)
    print("Gripper set to closed position.")
    time.sleep(2.0)


def open_door_sequence(robot: Robot, target: DoorTarget, handle_pose: Pose3D, door_type: str,
                       dist: float = HANDLE_DISTANCE, roll: float = 0.0) -> bool:
    """
    Close in on the handle, step to the door's swing side, grab the knob and pull it open
    """
    
    if door_type not in ("left", "right"):
        print(f"Door {target.container_id} reports type '{door_type}', not a hinged door.")
        return False

    print(f"----------------- {door_type.upper()} DOOR OPENING -----------------")
    body_pose, _ = plan_body_pose(searchnet_planning.plan_door_search, target, handle_pose, dist)

    print("-----Moving in front of the door before opening...-----")
    move_in_front_of(robot.stow, robot.base, robot.head, robot.pose, robot.tf, body_pose,
                     handle_pose, 0.0, 0.0, 0.0, 0.04, stow=False, grasp=True)

    beside_pose = body_pose_beside_door(body_pose, handle_pose, door_type)
    move_in_side_of(robot.stow, robot.base, robot.head, robot.pose, beside_pose, handle_pose,
                    robot.tf, 0.0, 0.0, 0.0, 0.05, stow=True, grasp=True, small=False)
    print(f"Executing door opening for {target.furniture_name}")
    time.sleep(5.0)

    grab_knob(robot, handle_pose, roll=roll)
    open_door(robot.pose)
    print("Door opening motion executed.")

    return_to_opened_door(robot, target, handle_pose, door_type, dist=dist/2)


    center_on_object(robot, "toy cat")  # center on the object in the opened door
    #detect object in opened door
    time.sleep(3.0)
    servo_grasp_object(pose_node=robot.pose, switch_mode=False, object_name="toy cat")
    return True


def center_on_object(robot: Robot, object_name: str):
    
    """?
    After opening the door and returning to position in front of it,
    run detection of the object with the gripper camera, get the 3D center, 
    and pass to correct_offsets_base_frame to center the robot.
    """
    print(f"Centering on {object_name} using gripper camera...")
    
    # 1. Capture RGB and Depth from the gripper camera
    rgb_img, depth_img = capture_gripper_view(robot)
    
    # 2. Run SAM3 detection
    detected, detection_dict = sam3_detect_object(object_name, rgb_img)
    if not detected:
        print(f"Could not find '{object_name}' in the opened door.")
        return
        
    # 3. Get the 2D bounding box center
    bb = detection_dict['box'] # (xmin, ymin, xmax, ymax)
    x_px = int((bb[0] + bb[2]) // 2)
    y_px = int((bb[1] + bb[3]) // 2)
    
    # 4. Unproject 2D pixel to 3D gripper camera frame
    camera_matrix = intrinsics_from_camera('/gripper_camera/color/camera_info')
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    
    z = robust_depth_m(depth_img, x_px, y_px)
    if z <= 0:
        print(f"Invalid depth at '{object_name}' center.")
        return
        
    x = (x_px - cx) * z / fx
    y = (y_px - cy) * z / fy
    object_pose_gripper = np.array((x, y, z, 1.0))
    
    # 5. Transform into the map frame and populate coords_cam (required by correct_offsets_base_frame)
    tf = robot.tf.get_tf_matrix("map", "gripper_camera_color_optical_frame")
    spin_until_complete(robot.tf)
    
    object_pose = Pose3D((tf @ object_pose_gripper)[:3])
    object_pose.coords_cam = object_pose_gripper[:3].copy()
    
    # 6. Correct offsets
    correct_offsets_base_frame(robot.tf, robot.pose, object_pose)
    

def return_to_opened_door(robot: Robot, target: DoorTarget, handle_pose: Pose3D, door_type: str,
                       dist: float = HANDLE_DISTANCE/2) -> None:
    """
    After opening the door, return to a position in front of it for further actions.
    """

    body_pose, _ = plan_body_pose(searchnet_planning.plan_door_search, target, handle_pose, dist)
    move_in_front_of(robot.stow, robot.base, robot.head, robot.pose, robot.tf, body_pose,
                     handle_pose, 0.0, 0.0, 0.0, 0.04, stow=False, grasp=True)
    print(f"Returned to position in front of the opened door {target.furniture_name}.")