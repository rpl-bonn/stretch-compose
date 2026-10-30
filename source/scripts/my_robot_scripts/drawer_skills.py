
import json
import os
import time
from dataclasses import dataclass

import numpy as np

from utils.robot_utils import searchnet_planning
from utils.robot_utils.robot_session import Robot
from utils.robot_utils.run_visual_servo import servo_grasp_object
from stretch_package.stretch_images.aligned_depth2color_subscriber import AlignedDepth2ColorSubscriber
from stretch_package.stretch_images.rgb_image_subscriber import RGBImageSubscriber
from utils.coordinates import Pose3D
from utils.recursive_config import Config
from utils.robot_utils.advanced_movement import (correct_offsets_base_frame, look_into_drawer,
                                                 move_in_front_of, pull_and_check_drawer, push)
from utils.robot_utils.basic_perception import get_depth_picture, get_rgb_picture
from utils.robot_utils.visual_servo_perception import GRIPPER_COLOR_TOPIC, GRIPPER_DEPTH_TOPIC
from utils.zero_shot_object_detection_sam3 import detect_drawer_handle_sam3, detect_handle_simple

config = Config()
GRAPH_DIR = os.path.join(config.get_subpath("scene_graph"), config["pre_scanned_graphs"]["high_res"])
IMG_DIR = config.get_subpath("images")

PROMPTS = ["drawer", "handle"]
CLOSE_PROMPTS = ["handle", "circle"]

# How far in front of the furniture to stand
APPROACH_DISTANCE = 0.7
#closer is better so that arm can reach inside the drawer
CENTERING_DISTANCE = 0.5


@dataclass
class DrawerTarget:
    """Everything the scene graph knows about one drawer"""

    drawer_id: int
    center: Pose3D
    furniture_id: str
    furniture_center: np.ndarray
    furniture_name: str


def load_scene(graph_dir: str = GRAPH_DIR) -> tuple[dict, dict]:
    """
    Read the scene graph and the furniture scene.
    """
    with open(os.path.join(graph_dir, "graph.json"), "r") as file:
        graph_data = json.load(file)
    with open(os.path.join(graph_dir, "scene.json"), "r") as file:
        scene_data = json.load(file)
    return graph_data, scene_data


def load_drawer_target(drawer_id: int, graph_data: dict, scene_data: dict,
                       graph_dir: str = GRAPH_DIR) -> DrawerTarget:
    """
    Resolve a drawer id to its centroid and the furniture it belongs to.
    """

    with open(os.path.join(graph_dir, "drawers", f"{drawer_id}.json"), "r") as file:
        drawer_data = json.load(file)
    furniture_id = str(graph_data["connections"][str(drawer_id)])
    furniture = scene_data["furniture"][furniture_id]
    return DrawerTarget(drawer_id, Pose3D(np.array(drawer_data["centroid"])), furniture_id,
                        furniture["centroid"], furniture["label"])


def capture_gripper_view(robot: Robot) -> tuple[np.ndarray, np.ndarray]:
    rgb = get_rgb_picture(RGBImageSubscriber, robot.pose, GRIPPER_COLOR_TOPIC, gripper=True)
    depth = get_depth_picture(AlignedDepth2ColorSubscriber, robot.pose, GRIPPER_DEPTH_TOPIC,
                              gripper=True)
    return rgb, depth


def approach_drawer(robot: Robot, target: DrawerTarget, dist: float = APPROACH_DISTANCE,
                    attempts: int = 5, tolerance: float = 0.5) -> np.ndarray:
    """
    Drive the robot in front of the drawer
    """

    body_pose = None
    front_normal = None
    for attempt in range(attempts):
        body_pose, front_normal = searchnet_planning.plan_drawer_search(
            target.furniture_name, target.furniture_center, target.center, dist,
            tolerance=tolerance + 0.1 * attempt)
        if body_pose is not None:
            break
        print(f"[approach] no body pose for drawer {target.drawer_id}, "
              f"retry {attempt + 1}/{attempts} with a wider tolerance")
    if body_pose is None or front_normal is None:
        raise RuntimeError(f"Could not plan an approach to drawer {target.drawer_id}.")

    print(f"Body pose: {body_pose}, front normal: {front_normal}")
    move_in_front_of(robot.stow, robot.base, robot.head, robot.pose, robot.tf, body_pose,
                     target.center, 0.0, 0.0, 0.0, 0.09, stow=True, grasp=True)
    return front_normal


def detect_handle(robot: Robot, target: DrawerTarget, front_normal: np.ndarray,
                  prompts: list = PROMPTS) -> tuple[Pose3D, str]:
 
    handle_pose = None
    drawer_type = None
    while handle_pose is None:
        rgb_img, depth_img = capture_gripper_view(robot)
        handle_pose, drawer_type, _ = detect_drawer_handle_sam3(
            robot.tf, depth_img, rgb_img, prompts, target_pos=target.center.coordinates)
        if handle_pose is None:
            print("No handle detected, retrying...")

    print(f"HANDLE POSE: {handle_pose}")
    print(f"DRAWER TYPE: {drawer_type}")
    handle_pose.set_rot_from_direction(-front_normal)
    return handle_pose, drawer_type


def center_on_handle(robot: Robot, target: DrawerTarget, handle_pose: Pose3D,
                     dist: float = CENTERING_DISTANCE, prompts: list = PROMPTS) -> Pose3D:
    """
    Close in on the handle and line the gripper up with it.
    """

    body_pose, _ = searchnet_planning.plan_drawer_search(
        target.furniture_name, target.furniture_center, handle_pose, dist)
    if body_pose is None:
        raise RuntimeError(f"Could not plan a centering pose for drawer {target.drawer_id}.")
    move_in_front_of(robot.stow, robot.base, robot.head, robot.pose, robot.tf, body_pose,
                     handle_pose, 0.0, 0.0, np.pi / 2, 0.09, stow=False, grasp=True)

    handle_pose_refine = None
    while handle_pose_refine is None:
        rgb_img, depth_img = capture_gripper_view(robot)
        handle_pose_refine, _, _ = detect_drawer_handle_sam3(
            robot.tf, depth_img, rgb_img, prompts, target_pos=target.center.coordinates)
        if handle_pose_refine is None:
            print("Re-detection of handle failed, retrying...")

    print(f"[centering] refreshed handle_pose: {handle_pose_refine.coordinates} "
          f"(was {handle_pose.coordinates})")
    handle_pose = handle_pose_refine

    time.sleep(3.0)
    correct_offsets_base_frame(robot.tf, robot.pose, handle_pose)
    return handle_pose


def measure_handle(robot: Robot, prompts: list = CLOSE_PROMPTS, attempts: int = 1) -> Pose3D | None:
    """
    Measure the handle from where the gripper is standing right now
    """

    for attempt in range(attempts):
        rgb_close, depth_close = capture_gripper_view(robot)
        handle_pose = detect_handle_simple(robot.tf, depth_close, rgb_close, prompts)
        if handle_pose is not None:
            return handle_pose
        print(f"[measure] no handle in view (attempt {attempt + 1}/{attempts})")
    return None


def handle_pose_at_height(height: float) -> Pose3D:
    return Pose3D(np.array([0.0, 0.0, height]))


def grasp_from_drawer(robot: Robot, handle_pose: Pose3D | None = None,
                      object_name: str | None = None, attempts: int = 3) -> bool:
    """
    Pull the drawer open, look inside, and servo the gripper onto the object.
    """

    if handle_pose is None:
        handle_pose = measure_handle(robot, attempts=attempts)
        if handle_pose is None:
            print("[grasp] no handle to work from; park the gripper in front of one, or pass "
                  "handle_pose_at_height(<metres>).")
            return False
        print(f"[grasp] measured handle at {handle_pose.coordinates.round(3)}")

    if not robot.switch_to_position_mode():
        print("[grasp] could not switch to position mode; is stretch_driver running?")
        return False

    if not pull_and_check_drawer(robot.pose, robot.tf, lambda: measure_handle(robot), IMG_DIR=IMG_DIR):
        print("[grasp] drawer was not pulled open; nothing to grasp from.")
        return False
    time.sleep(2.0)
    grasped=True
    look_into_drawer(robot.pose, handle_pose)
    time.sleep(5.0)

    servo_kwargs = {} if object_name is None else {'object_name': object_name}
    grasped = servo_grasp_object(pose_node=robot.pose, switch_mode=False, **servo_kwargs)
    print(f"Visual servo grasp {'succeeded' if grasped else 'failed'}.")
    return grasped


def close_drawer(robot: Robot, handle_pose: Pose3D, clearance: float = 0.09) -> None:
    """
    Push the drawer shut.

    Args:
        robot (Robot): session handles
        handle_pose (Pose3D): the handle pose, whose height the push is aimed below
        clearance (float): how far below the handle to push, so the gripper clears it
    """

    #retract first 
    

    time.sleep(1.0)
    push(robot.pose, handle_pose.coordinates[2] - clearance)
