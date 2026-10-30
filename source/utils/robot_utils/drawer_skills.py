"""
The steps that are a drawer's own: pull it straight out, look inside, grasp, push it shut.

Everything up to the handle - loading the scene, driving to the container, detecting and
measuring the handle - is shared with doors and lives in container_skills. This module re-exports
those under drawer names so a drawer demo reads as one sequence.

The usual order:

    target       = load_drawer_target(drawer_id, *load_scene())
    front_normal = approach_drawer(robot, target)
    handle_pose, drawer_type = detect_handle(robot, target, front_normal)
    handle_pose  = center_on_handle(robot, target, handle_pose)      # front drawers only
    grasped      = grasp_from_drawer(robot, handle_pose)
    close_drawer(robot, handle_pose)
"""

import time

import numpy as np

from utils.robot_utils.run_visual_servo import servo_grasp_object
from utils.coordinates import Pose3D
from utils.robot_utils import searchnet_planning
from utils.robot_utils.advanced_movement import (correct_offsets_base_frame, look_into_drawer,
                                                 move_in_front_of, pull_and_check_drawer, push)
from utils.robot_utils.container_skills import (GRAPH_DIR, IMG_DIR, ContainerTarget,
                                                approach_container, capture_gripper_view,
                                                load_container_target, measure_handle,
                                                plan_body_pose)
# Re-exported so a drawer demo can pull its whole sequence from one module.
from utils.robot_utils.container_skills import handle_pose_at_height, load_scene  # noqa: F401
from utils.robot_utils.container_skills import detect_handle as detect_container_handle
from utils.robot_utils.robot_session import Robot
from utils.zero_shot_object_detection_sam3 import detect_drawer_handle_sam3

# A drawer target is a container target; the name is kept so drawer code reads as drawer code.
DrawerTarget = ContainerTarget

PROMPTS = ["drawer", "handle"]

# How far in front of the furniture to stand
APPROACH_DISTANCE = 0.7
#closer is better so that arm can reach inside the drawer
CENTERING_DISTANCE = 0.5


def load_drawer_target(drawer_id: int, graph_data: dict, scene_data: dict,
                       graph_dir: str = GRAPH_DIR) -> DrawerTarget:
    """
    Resolve a drawer id to its centroid and the furniture it belongs to.

    Args:
        drawer_id (int): id of the drawer in the scene graph
        graph_data (dict): graph.json contents
        scene_data (dict): scene.json contents
        graph_dir (str): directory holding the per-drawer json files

    Returns:
        DrawerTarget
    """
    return load_container_target(drawer_id, graph_data, scene_data, graph_dir)


def approach_drawer(robot: Robot, target: DrawerTarget, dist: float = APPROACH_DISTANCE,
                    attempts: int = 5, tolerance: float = 0.5) -> np.ndarray:
    """
    Drive the robot in front of the drawer.

    Args:
        robot (Robot): session handles
        target (DrawerTarget): the drawer to stand in front of
        dist (float): how far from the drawer to stand
        attempts (int): re-planning attempts
        tolerance (float): initial furniture match tolerance

    Returns:
        np.ndarray: the furniture's front normal
    """
    return approach_container(robot, target, searchnet_planning.plan_drawer_search, dist,
                              roll=0.0, lift=0.09, attempts=attempts, tolerance=tolerance)


def detect_handle(robot: Robot, target: DrawerTarget, front_normal: np.ndarray,
                  prompts: list = PROMPTS) -> tuple[Pose3D, str]:
    """
    Find the drawer's handle from the gripper camera, retrying until it is seen.

    Args:
        robot (Robot): session handles
        target (DrawerTarget): supplies the centroid the detection is matched against
        front_normal (np.ndarray): from approach_drawer
        prompts (list): SAM3 prompts

    Returns:
        tuple: (handle_pose in map, drawer_type - "front", "left" or "right")
    """
    return detect_container_handle(robot, target, front_normal, detect_drawer_handle_sam3, prompts)


def center_on_handle(robot: Robot, target: DrawerTarget, handle_pose: Pose3D,
                     dist: float = CENTERING_DISTANCE, prompts: list = PROMPTS) -> Pose3D:
    """
    Close in on the handle and line the gripper up with it.

    Re-plans against the handle rather than the drawer centroid, then re-detects from the new
    viewpoint: the offset correction has to work from a fresh measurement, not from a pose
    computed where the robot used to stand.

    Args:
        robot (Robot): session handles
        target (DrawerTarget): supplies the furniture to plan against
        handle_pose (Pose3D): handle pose from detect_handle
        dist (float): how far from the handle to stand
        prompts (list): SAM3 prompts for the re-detection

    Returns:
        Pose3D: the refreshed handle pose
    """
    body_pose, _ = plan_body_pose(searchnet_planning.plan_drawer_search, target, handle_pose, dist)
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


def grasp_from_drawer(robot: Robot, handle_pose: Pose3D | None = None,
                      object_name: str | None = None) -> bool:
    """
    Pull the drawer open, look inside, and servo the gripper onto the object.

    Args:
        robot (Robot): session handles
        handle_pose (Pose3D): the centred handle pose. Omit it to measure the handle from the
            current gripper view instead, which is what makes this step runnable on its own
            with the robot parked in front of an open drawer by hand
        object_name (str): SAM3 prompt for the object to grasp; the servo default if omitted

    Returns:
        bool: whether the object ended up held
    """
    if handle_pose is None:
        handle_pose = measure_handle(robot)
        print(f"[grasp] measured handle at {handle_pose.coordinates.round(3)}")

    # The pull, the look-in and the servo all need position mode, and nothing guarantees it when
    # this step runs on its own. Switched here rather than inside the servo so the arm moves
    # before it are covered too.
    if not robot.switch_to_position_mode():
        print("[grasp] could not switch to position mode; is stretch_driver running?")
        return False

    # Pull the drawer halfway, re-check the handle offset, correct the base if it has drifted.
    # The reach is measured from the handle each time, so standing closer changes the commanded
    # extension instead of driving the gripper into the drawer front.
    if not pull_and_check_drawer(robot.pose, robot.tf, lambda: measure_handle(robot), IMG_DIR=IMG_DIR):
        print("[grasp] drawer was not pulled open; nothing to grasp from.")
        return False
    time.sleep(2.0)
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
    time.sleep(1.0)
    push(robot.pose, handle_pose.coordinates[2] - clearance)
