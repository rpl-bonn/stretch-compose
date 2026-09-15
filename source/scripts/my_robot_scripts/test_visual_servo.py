"""
Script for testing the visual servoing loop on a single object
"""

import argparse
from threading import Thread

import numpy as np
import open3d as o3d
import rclpy
import tf2_ros
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from stretch_package.stretch_images.aligned_depth2color_subscriber import AlignedDepth2ColorSubscriber
from stretch_package.stretch_images.rgb_image_subscriber import RGBImageSubscriber
from stretch_package.stretch_movement.mode_controller import ModeController
from stretch_package.stretch_movement.move_to_pose import JointPoseController
from stretch_package.stretch_movement.move_to_position import JointPositionController
from stretch_package.stretch_movement.visual_servo import (
    GRIPPER_COLOR_TOPIC, GRIPPER_DEPTH_TOPIC, VisualServoController)
from stretch_package.stretch_state.frame_transformer import FrameTransformer
from utils.coordinates import Pose3D
from utils.point_clouds import collect_dynamic_point_cloud
from utils.recursive_config import Config
from utils.robot_utils.advanced_movement import find_new_grasp_dynamically, sam3_box_detector
from utils.robot_utils.basic_movement import set_gripper, spin_until_complete
from utils.robot_utils.basic_perception import get_depth_picture, get_rgb_picture
from utils.zero_shot_object_detection_sam3 import sam3_detect_object

OBJECT = "tennis ball"  
GRIPPER_CAMERA_FRAME = "gripper_camera_color_optical_frame"

# Arm pose that puts the gripper camera in front of the object with both finger markers in
# view. Tune the lift to your table height; extension stays short so the servo loop has room
# to extend. Set --no-pose to skip this and place the arm by hand instead.
READY_POSE = {
    # 'joint_lift': 0.144,
    'wrist_extension': 0.05,
    'joint_wrist_yaw': 0.0,
    'joint_wrist_pitch': -0.2,
    'joint_wrist_roll': 0.0,
}

# Gripper width to close to, and the width the grasp check sizes its aperture window from. A
# tennis ball is 6.6 cm; on the --gpd path GPD supplies this per object instead.
OBJECT_WIDTH_M = 0.066

# Distances find_new_grasp_dynamically approaches from and to, matching demo_grasping.
GRASP_DISTANCE_START = 0.15
GRASP_DISTANCE_END = 0.05


class TransformManager:
    """Spins a node so the TF buffer stays filled, as demo_grasping does."""

    def __init__(self, node: Node):
        self.node = node
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=100))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self._spin_thread = Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    def _spin(self):
        self.executor.spin()

    def shutdown(self):
        self.executor.shutdown()
        self._spin_thread.join()


def locate_object_in_map(servo_node, pose_node, transform_node, object_name):
    """
    Find the object with the gripper camera and express its grasp point in the map frame.

    This stands in for plan_object_search, which gets the object position from the head camera
    plus the scene graph. Here the gripper is already pointed at the object, so its own camera
    is the more direct source and no scene graph is needed.

    Args:
        servo_node (VisualServoController): supplies live intrinsics and the target reduction
        pose_node (JointPoseController): used by the image capture helpers
        transform_node (FrameTransformer): map <- camera lookup
        object_name (str): SAM3 text prompt

    Returns:
        tuple: (centre Pose3D in map, gripper Pose3D in map), or (None, None) if not detected
    """
    rgb = get_rgb_picture(RGBImageSubscriber, pose_node, GRIPPER_COLOR_TOPIC, gripper=True,
                          save_block=True)
    depth = get_depth_picture(AlignedDepth2ColorSubscriber, pose_node, GRIPPER_DEPTH_TOPIC,
                              gripper=True, save_block=True)
    detected, detection = sam3_detect_object(object_name, rgb)
    if not detected:
        print(f"'{object_name}' not visible from the gripper camera.")
        return None, None

    target = servo_node.perception.locate_target(depth, box=detection['box'])
    if target is None:
        print("Depth inside the detection box was unusable.")
        return None, None
    print(f"Object at {np.round(100 * target['xyz'], 1)} cm in the camera frame, "
          f"apparent width {100 * target['width_m']:.1f} cm.")

    camera_tform_map = transform_node.get_tf_matrix("map", GRIPPER_CAMERA_FRAME)
    centre_map = np.array([*target['xyz'], 1.0]) @ camera_tform_map.T

    gripper_tform_map = transform_node.get_tf_matrix("map", "link_grasp_center")
    gripper_map = np.array((0.0, 0.0, 0.0, 1.0)) @ gripper_tform_map.T

    # plan_object_search orients the centre along -front_normal, the direction the robot faces
    # the furniture from. Without a scene graph the equivalent is the horizontal direction from
    # the gripper to the object.
    approach = centre_map[:3] - gripper_map[:3]
    approach[2] = 0.0
    norm = np.linalg.norm(approach)
    centre = Pose3D(centre_map[:3])
    if norm > 1e-6:
        centre.set_rot_from_direction(approach / norm)

    start_pose = Pose3D(gripper_map[:3], centre.rot_matrix.copy())
    return centre, start_pose


def run_gpd_grasp(args, servo_node, pose_node, position_node, transform_node):
    """
    Detect, build the dynamic point cloud, plan with GPD, then grasp with servoing.

    Args:
        args: parsed arguments
        servo_node (VisualServoController): closes the loop on the final approach
        pose_node (JointPoseController): gripper and wrist commands
        position_node (JointPositionController): cartesian arm moves
        transform_node (FrameTransformer): frame lookups

    Returns:
        bool: whether the object ended up held
    """
    centre, start_pose = locate_object_in_map(servo_node, pose_node, transform_node, args.object)
    if centre is None:
        return False
    print(f"Object centre in map: {np.round(centre.coordinates, 3)}")

    # full_env_pcd is only used by commented-out visualisation inside collect_dynamic_point_cloud,
    # but it is a required argument, so hand over the same scene cloud demo_grasping uses.
    config = Config()
    scene_path = f"{config.get_subpath('aligned_point_clouds')}/{config['pre_scanned_graphs']['high_res']}/scene.ply"
    scene_pcd = o3d.io.read_point_cloud(scene_path)

    print("Sweeping arc views to build the object point cloud...")
    pcd_obj, pcd_env = collect_dynamic_point_cloud(
        args.object, position_node, pose_node, transform_node, start_pose, centre, scene_pcd)
    print(f"Object cloud: {len(pcd_obj.points)} points, environment cloud: {len(pcd_env.points)}")

    # body_pose is accepted but unused by find_new_grasp_dynamically; pass the centre so the
    # argument at least means something if that ever changes.
    return find_new_grasp_dynamically(
        position_node, pose_node, transform_node, centre,
        GRASP_DISTANCE_START, GRASP_DISTANCE_END, config, pcd_obj, pcd_env,
        servo_node=servo_node if not args.no_servo else None, servo_object=args.object)


def run_servo_only(servo_node, pose_node, object_name=OBJECT, timeout_s=30.0, grasp=True,
                   dry_run=False, flat=False, width=OBJECT_WIDTH_M, return_to_ready=False):
    """
    Servo straight onto the object, with no grasp planning.

    Args:
        servo_node (VisualServoController): the loop under test
        pose_node (JointPoseController): gripper commands
        object_name (str): SAM3 text prompt
        timeout_s (float): servo timeout
        grasp (bool): close the gripper afterwards and verify the grasp
        dry_run (bool): perceive and log only, never move
        flat (bool): object lies flat, aim at the visible surface instead of an assumed centre
        width (float): object width in meters, sizes the grasp confirmation window
        return_to_ready (bool): send the arm back to READY_POSE once the object is held

    Returns:
        bool: whether the grasp was confirmed, or whether servoing converged if grasp is off
    """
    print(f"Servoing to '{object_name}'{' (dry run)' if dry_run else ''}...")
    result = servo_node.servo_to_object(
        object_name,
        sam3_box_detector,
        timeout_s=timeout_s,
        push_to_center=not flat,
        dry_run=dry_run,
        debug_image_path='/home/ws/data/images/visual_servo_last_frame.png',
    )
    print(f"\n{result}")
    print("Last annotated frame: /home/ws/data/images/visual_servo_last_frame.png")

    # if not (grasp and result.success):
    #     return result.success

    print("Closing gripper.")
    set_gripper(pose_node, -0.35)
    verification = servo_node.verify_grasp(expected_width_m=width)
    print(f"{verification}")
    print(f"Grasp {'CONFIRMED' if verification.grasped else 'NOT confirmed'}.")
    if return_to_ready:
        pose_node.send_joint_pose(READY_POSE)
        spin_until_complete(pose_node)
    return verification.grasped


def servo_grasp_object(object_name=OBJECT, pose_node=None, velocity_scale=0.6,
                       switch_mode=True, **kwargs):
    """
    Run the servo-and-grasp sequence inside a pipeline that is already running rclpy.

    This is what a demo script should call instead of shelling out to this file: it borrows the
    caller's context and nodes, so no second rclpy context, no duplicate camera subscriptions,
    and the result comes back as a bool rather than a process exit code. The servo node spins
    itself with spin_once, so it must not be added to the caller's executor.

    Args:
        object_name (str): SAM3 text prompt
        pose_node (JointPoseController): the caller's pose node; one is created if omitted
        velocity_scale (float): overall velocity scale for the servo loop
        switch_mode (bool): switch the driver to position mode first. Skip it if the caller
            already did, since the switch interrupts whatever the arm is doing.
        **kwargs: forwarded to run_servo_only (timeout_s, grasp, flat, width, dry_run,
            return_to_ready)

    Returns:
        bool: whether the object ended up held
    """
    if not rclpy.ok():
        raise RuntimeError("rclpy is not initialised; call this from a running pipeline.")

    owned = []
    servo_node = VisualServoController(velocity_scale=velocity_scale)
    owned.append(servo_node)
    try:
        if pose_node is None:
            pose_node = JointPoseController()
            owned.append(pose_node)
        if switch_mode:
            mode_node = ModeController()
            owned.append(mode_node)
            if not mode_node.switch_to_position_mode():
                print("Could not switch to position mode; is stretch_driver running?")
                return False

        # The servo loop refuses to approach with half-closed fingers, and an open gripper is
        # what puts both finger markers in the camera's view.
        set_gripper(pose_node, True)
        return run_servo_only(servo_node, pose_node, object_name=object_name, **kwargs)
    finally:
        for node in owned:
            node.destroy_node()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--object', default=OBJECT, help='SAM3 text prompt')
    parser.add_argument('--gpd', action='store_true',
                        help='run the full detect -> point cloud -> GPD -> grasp pipeline')
    parser.add_argument('--dry-run', action='store_true',
                        help='perceive and log the error without moving the robot (servo only)')
    parser.add_argument('--scale', type=float, default=0.6,
                        help='overall velocity scale; 1.0 uses the full per-joint ceilings')
    parser.add_argument('--timeout', type=float, default=30.0, help='servo timeout in seconds')
    parser.add_argument('--grasp', action='store_true', default=True,
                        help='close the gripper afterwards and verify the grasp (servo only)')
    parser.add_argument('--no-servo', action='store_true',
                        help='with --gpd, grasp open loop so GPD can be judged on its own')
    parser.add_argument('--flat', action='store_true',
                        help='object lies flat on a surface: aim at the visible surface instead '
                             'of offsetting towards an assumed centre')
    parser.add_argument('--no-pose', action='store_true',
                        help='skip moving the arm to READY_POSE')
    parser.add_argument('--width', type=float, default=OBJECT_WIDTH_M,
                        help='object width in meters, sizes the grasp confirmation window')
    args = parser.parse_args()

    rclpy.init(args=None)
    transform_manager = None
    position_node = None
    transform_node = None
    pose_node = JointPoseController()
    servo_node = VisualServoController(velocity_scale=args.scale)
    mode_node = ModeController()

    try:
        if not mode_node.switch_to_position_mode():
            print("Could not switch to position mode; is stretch_driver running?")
            return

        if not args.no_pose:
            print(f"Moving arm to the ready pose: {READY_POSE}")
            set_gripper(pose_node, True)
            # pose_node.send_joint_pose(READY_POSE)
            # spin_until_complete(pose_node)

        # The servo loop refuses to approach with half-closed fingers, and an open gripper is
        # what puts both finger markers in the camera's view.
        print("Opening gripper.")
        set_gripper(pose_node, True)

        if args.gpd:
            # Only the GPD path needs TF, so the servo-only path stays free of it.
            transform_manager = TransformManager(rclpy.create_node('servo_test_tf_node'))
            position_node = JointPositionController(transform_manager.tf_buffer)
            transform_node = FrameTransformer(transform_manager.tf_buffer)
            if not servo_node._wait_for_inputs():
                return
            success = run_gpd_grasp(args, servo_node, pose_node, position_node, transform_node)
        else:
            success = run_servo_only(servo_node, pose_node, object_name=args.object,
                                     timeout_s=args.timeout, grasp=args.grasp,
                                     dry_run=args.dry_run, flat=args.flat, width=args.width)

        print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        if transform_manager is not None:
            transform_manager.shutdown()
        for node in (position_node, transform_node, pose_node, servo_node, mode_node):
            if node is not None:
                node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
