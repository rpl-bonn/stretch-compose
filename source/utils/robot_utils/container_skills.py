"""
Helper functions for the container-opening demos, which are shared between drawers and doors
"""

import json
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from stretch_package.stretch_images.aligned_depth2color_subscriber import AlignedDepth2ColorSubscriber
from stretch_package.stretch_images.rgb_image_subscriber import RGBImageSubscriber
from utils.coordinates import Pose3D
from utils.recursive_config import Config
from utils.robot_utils.advanced_movement import move_in_front_of
from utils.robot_utils.basic_perception import get_depth_picture, get_rgb_picture
from utils.robot_utils.robot_session import Robot
from utils.robot_utils.visual_servo_perception import GRIPPER_COLOR_TOPIC, GRIPPER_DEPTH_TOPIC
from utils.zero_shot_object_detection_sam3 import detect_handle_simple

config = Config()
GRAPH_DIR = os.path.join(config.get_subpath("scene_graph"), config["pre_scanned_graphs"]["high_res"])
IMG_DIR = config.get_subpath("images")

CLOSE_PROMPTS = ["handle", "circle"]


@dataclass
class ContainerTarget:
    """
    Everything the scene graph knows about one drawer or door
    """

    container_id: int
    center: Pose3D
    furniture_id: str
    furniture_center: np.ndarray
    furniture_name: str


def load_scene(graph_dir: str = GRAPH_DIR) -> tuple[dict, dict]:
    """
    Read the scene graph and the furniture scene
    """

    with open(os.path.join(graph_dir, "graph.json"), "r") as file:
        graph_data = json.load(file)
    with open(os.path.join(graph_dir, "scene.json"), "r") as file:
        scene_data = json.load(file)
    return graph_data, scene_data


def load_container_target(container_id: int, graph_data: dict, scene_data: dict,
                          graph_dir: str = GRAPH_DIR) -> ContainerTarget:
    """
    Resolve a drawer or door id to its centroid and the furniture it belongs to
    """

    with open(os.path.join(graph_dir, "drawers", f"{container_id}.json"), "r") as file:
        container_data = json.load(file)
    furniture_id = str(graph_data["connections"][str(container_id)])
    furniture = scene_data["furniture"][furniture_id]
    return ContainerTarget(container_id, Pose3D(np.array(container_data["centroid"])),
                           furniture_id, furniture["centroid"], furniture["label"])


def capture_gripper_view(robot: Robot) -> tuple[np.ndarray, np.ndarray]:
    """
    Take a matched colour/depth pair from the gripper camera
    """

    rgb = get_rgb_picture(RGBImageSubscriber, robot.pose, GRIPPER_COLOR_TOPIC, gripper=True)
    depth = get_depth_picture(AlignedDepth2ColorSubscriber, robot.pose, GRIPPER_DEPTH_TOPIC,
                              gripper=True)
    return rgb, depth


def plan_body_pose(planner: Callable, target: ContainerTarget, goal: Pose3D, dist: float,
                   attempts: int = 5, tolerance: float = 0.5) -> tuple[Pose3D, np.ndarray]:
    """
    Ask a searchnet planner where to stand, widening the furniture match on each retry
    """

    for attempt in range(attempts):
        body_pose, front_normal = planner(target.furniture_name, target.furniture_center, goal,
                                          dist, tolerance=tolerance + 0.1 * attempt)
        if body_pose is not None and front_normal is not None:
            print(f"Body pose: {body_pose}, front normal: {front_normal}")
            return body_pose, front_normal
        print(f"[approach] no body pose for container {target.container_id}, "
              f"retry {attempt + 1}/{attempts} with a wider tolerance")
    raise RuntimeError(f"Could not plan an approach to container {target.container_id}.")


def approach_container(robot: Robot, target: ContainerTarget, planner: Callable, dist: float,
                       roll: float = 0.0, lift: float = 0.09, attempts: int = 5,
                       tolerance: float = 0.5) -> np.ndarray:
    """
    Drive the robot in front of the container, gripper facing it
    """

    body_pose, front_normal = plan_body_pose(planner, target, target.center, dist,
                                             attempts=attempts, tolerance=tolerance)
    move_in_front_of(robot.stow, robot.base, robot.head, robot.pose, robot.tf, body_pose,
                     target.center, 0.0, 0.0, roll, lift, stow=True, grasp=True)
    return front_normal


def detect_handle(robot: Robot, target: ContainerTarget, front_normal: np.ndarray,
                  detector: Callable, prompts: list) -> tuple[Pose3D, str]:
    """
    Find the container's handle from the gripper camera, retrying until it is seen
    """

    handle_pose = None
    container_type = None
    while handle_pose is None:
        rgb_img, depth_img = capture_gripper_view(robot)
        handle_pose, container_type, _ = detector(robot.tf, depth_img, rgb_img, prompts,
                                                  target_pos=target.center.coordinates)
        if handle_pose is None:
            print("No handle detected, retrying...")

    print(f"HANDLE POSE: {handle_pose}")
    print(f"CONTAINER TYPE: {container_type}")
    handle_pose.set_rot_from_direction(-front_normal)
    return handle_pose, container_type


def measure_handle(robot: Robot, prompts: list = CLOSE_PROMPTS, frame: str = "map",
                   attempts: int | None = None, expected=None, max_dist: float = 0.10,
                   ambiguity_ratio: float | None = None, drop_cut_off: bool = False,
                   shape: str | None = None, neighbours=None) -> Pose3D | None:
    """
    Measure the handle from where the gripper is standing right now, retrying until it is seen.
    With attempts set, gives up and returns None after that many failed detections. With
    expected (a point in `frame`), only the handle nearest to it, within max_dist, counts, and
    with ambiguity_ratio only if it is that many times nearer than any other.
    """

    handle_pose = None
    tries = 0
    while handle_pose is None and (attempts is None or tries < attempts):
        rgb_close, depth_close = capture_gripper_view(robot)
        handle_pose = detect_handle_simple(robot.tf, depth_close, rgb_close, prompts, frame=frame,
                                           expected=expected, max_dist=max_dist,
                                           ambiguity_ratio=ambiguity_ratio, drop_cut_off=drop_cut_off,
                                           shape=shape, neighbours=neighbours)
        tries += 1
        if handle_pose is None:
            print("No handle detected, retrying...")
    return handle_pose


def handle_pose_at_height(height: float) -> Pose3D:
    return Pose3D(np.array([0.0, 0.0, height]))
