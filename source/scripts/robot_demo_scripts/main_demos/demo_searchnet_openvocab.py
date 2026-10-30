"""
Search for an object: sweep open shelves first, then check inside drawers and doors.
"""

import os
import subprocess
import time
import traceback

import cv2
import rclpy

from stretch_package.stretch_images.aligned_depth2color_subscriber import AlignedDepth2ColorSubscriber
from stretch_package.stretch_images.rgb_image_subscriber import RGBImageSubscriber
from utils.robot_utils import searchnet_planning
from utils.robot_utils.advanced_movement import look_into_drawer, move_in_front_of, pull_and_check_drawer, push
from utils.robot_utils.basic_movement import set_gripper, spin_until_complete
from utils.robot_utils.basic_perception import get_depth_picture, get_rgb_picture
from utils.robot_utils.container_skills import IMG_DIR, measure_handle
from utils.robot_utils.drawer_skills import (approach_drawer, center_on_handle, detect_handle,
                                            load_drawer_target, load_scene)
from utils.robot_utils.robot_session import Robot, robot_session
from utils.robot_utils.visual_servo_perception import GRIPPER_COLOR_TOPIC
from utils.zero_shot_object_detection_sam3 import sam3_detect_object

# Adaptable
VIS_BLOCK = False
SAVE_BLOCK = True
OBJECT = "ball"
OBJECT_DE = "Ball"
ROBOT_SSH = "hello-robot@192.168.50.34"

# Head pan/tilt used to sweep a shelf while scanning for the object (pan is positive = left)
HEAD_TILT = -0.2
HEAD_PAN_SWEEP = [0.0, 0.3, 0.6]
DRAWER_IDS_TO_CHECK = [9, 10, 11, 12]


def say_on_robot(text: str) -> None:
    """
    Have the robot speak a status update out loud over ssh
    """
    subprocess.run(["ssh", ROBOT_SSH, "espeak  -v de+f3 -s 110 -p 40", f'"{text}"'], timeout=20)


def search_open_spaces(robot: Robot, scene_data: dict, obj: str, obj_de: str) -> tuple[bool, dict]:
    """
    Sweep the bookshelf with the head camera, panning across it until the object is detected
    """

    print("Checking tables with SAM3...")
    detected, detection_dict = False, {}
    checked_furniture_ids = []
    bookshelf_plans = searchnet_planning.plan_furniture_search_by_label("bookshelf", scene_data)
    for target_pos, furniture, front_normal, body_pose, furniture_id in bookshelf_plans:
        if furniture_id in checked_furniture_ids:
            print(f"Already checked bookshelf ({furniture_id}).")
            continue
        checked_furniture_ids.append(furniture_id)
        print(f"Searching for {obj} on {furniture} ({furniture_id}) at {target_pos}")
        move_in_front_of(robot.stow, robot.base, robot.head, robot.pose, robot.tf, body_pose,
                         target_pos, 0.0, 0.0, 0.0, 0.0, stow=True, grasp=False)

        # Sweep head pan until the object is detected or all angles are exhausted
        for pan_angle in HEAD_PAN_SWEEP:
            print(f"Scanning shelf at pan={pan_angle:.2f} rad")
            robot.pose.send_joint_pose({'joint_head_pan': pan_angle, 'joint_head_tilt': HEAD_TILT})
            spin_until_complete(robot.pose)
            time.sleep(2.0)  # Let the images catch up with the new head pose

            rgb_img = get_rgb_picture(RGBImageSubscriber, robot.pose, '/camera/color/image_raw',
                                      gripper=False, save_block=SAVE_BLOCK, vis_block=VIS_BLOCK)
            get_depth_picture(AlignedDepth2ColorSubscriber, robot.pose,
                              '/camera/aligned_depth_to_color/image_raw', gripper=False,
                              save_block=SAVE_BLOCK, vis_block=VIS_BLOCK)
            detected, detection_dict = sam3_detect_object(obj, rgb_img, save_block=SAVE_BLOCK)
            time.sleep(2.0)  # Let SAM3 finish before moving on
            if detected:
                print(f"Found {obj} at head pan={pan_angle:.2f} rad")
                break

        if detected:
            say_on_robot(f"{obj_de} gefunden auf {furniture}")
        else:
            say_on_robot(f"{obj_de} nicht gefunden auf {furniture}. Ich überprüfe jetzt geschlossene Räume.")
        # Only the first matching furniture is ever checked; kept as-is from the original demo.
        break

    return detected, detection_dict


def search_concealed_spaces(robot: Robot, graph_data: dict, scene_data: dict, drawer_ids: list[int],
                            obj: str, obj_de: str) -> bool:
    """
    Check each drawer/door in turn
    """

    print(f"Object {obj} not found in open spaces. Searching in concealed spaces...")

    for drawer_id in drawer_ids:
        try:
            target = load_drawer_target(drawer_id, graph_data, scene_data)
            front_normal = approach_drawer(robot, target, dist=0.8)
            handle_pose, drawer_type = detect_handle(robot, target, front_normal)
        except Exception as e:
            print(f"[search] could not open drawer {drawer_id}: {e}")
            continue
        print(f"OPENING DIRECTION: {drawer_type}")

        if drawer_type == "front":
            handle_pose = center_on_handle(robot, target, handle_pose, dist=0.6)
            pull_and_check_drawer(robot.pose, robot.tf, lambda: measure_handle(robot), IMG_DIR=IMG_DIR)
            look_into_drawer(robot.pose, handle_pose)
            time.sleep(10.0)  # Give the object time to settle into view
            rgb_img = get_rgb_picture(RGBImageSubscriber, robot.pose, GRIPPER_COLOR_TOPIC,
                                      gripper=True, save_block=SAVE_BLOCK, vis_block=VIS_BLOCK)
        elif drawer_type in ("left", "right"):
            time.sleep(3.0)  # Wait for manual opening
            set_gripper(robot.pose, True)
            rgb_img = get_rgb_picture(RGBImageSubscriber, robot.pose, GRIPPER_COLOR_TOPIC,
                                      gripper=True, save_block=SAVE_BLOCK, vis_block=VIS_BLOCK)
        else:
            print(f"Container {drawer_id} reports type '{drawer_type}', skipping.")
            continue

        detected, detection_dict = sam3_detect_object(obj, rgb_img, conf=0.2, save_block=SAVE_BLOCK)

        if detected:
            print(f"Found {obj} in door {drawer_id} in {target.furniture_name}.")
            time.sleep(5.0)
            say_on_robot(f"{obj_de} gefunden in {target.furniture_name} in der Tür.")
        else:
            print(f"{obj} not found in {target.furniture_name} in drawer {drawer_id}.")
            say_on_robot(f"{obj_de} nicht gefunden in {target.furniture_name} in der Tür. "
                        f"Ich überprüfe den nächsten möglichen Ort.")

        # Close it back up before moving on, regardless of the result
        time.sleep(5.0)
        push(robot.pose, handle_pose.coordinates[2] - 0.12)
        time.sleep(2.0)

        if detected:
            return True

    return False


def execute_search(obj: str = OBJECT, obj_de: str = OBJECT_DE) -> bool:
    """
    Search for an object: sweep open shelves first, then check inside drawers and doors.

    Args:
        obj (str): object to search for
        obj_de (str): German name of the object, used for the spoken status updates

    Returns:
        bool: True if the object was found, False otherwise
    """
    try:
        with robot_session() as robot:
            graph_data, scene_data = load_scene()
            say_on_robot(f"Hallo! Suche nach {obj_de} gestartet.")

            detected, _ = search_open_spaces(robot, scene_data, obj, obj_de)
            if detected:
                return True

            return search_concealed_spaces(robot, graph_data, scene_data, DRAWER_IDS_TO_CHECK, obj, obj_de)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_sam3_detection(obj: str, capture_image: bool = True):
    """
    Test SAM3 object detection.
    """
    saved_image_path = os.path.join(IMG_DIR, "gripper_image_rgb.png")
    rgb_img = None

    rclpy.init()
    try:
        if capture_image:
            print(f"Capturing gripper image for SAM3 test...")
            # Instantiate directly to avoid set_gripper call in get_rgb_picture
            image_node = RGBImageSubscriber("/gripper_camera/color/image_rect_raw", rotate=False, save=True, vis=VIS_BLOCK)
            rgb_img = image_node.cv_image
            image_node.destroy_node()

        if rgb_img is None:
            if not os.path.exists(saved_image_path):
                print(f"No saved image found at {saved_image_path}. Run with capture_image=True first.")
                return False, {}
            print(f"Reusing saved image at {saved_image_path}")
            rgb_img = cv2.imread(saved_image_path)

        print(f"Running SAM3 detection for '{obj}'...")
        detected, detection_dict = sam3_detect_object(obj, rgb_img, conf=0.2, save_block=SAVE_BLOCK)
        print(f"Detection result for {obj}: {detected}, {detection_dict}")
        return detected, detection_dict
    finally:
        rclpy.shutdown()
     
if __name__ == "__main__":
    # docker run --net=bridge --mac-address=02:42:ac:11:00:02 -p 5000:5000 --gpus all -it craiden/graspnet:v1.0 python3 app.py
    # docker run -p 5004:5004 --gpus all -it craiden/yolodrawer:v1.0 python3 app.py
    # ./run_docker_new.sh
    execute_search(OBJECT, OBJECT_DE)
    # detected, detection_dict = test_sam3_detection(OBJECT)