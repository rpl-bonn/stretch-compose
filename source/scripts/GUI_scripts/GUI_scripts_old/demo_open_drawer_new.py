#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
import os
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
import tf2_ros
from threading import Thread
import traceback

from scripts.my_robot_scripts import searchnet_planning
from stretch_package.stretch_images.aligned_depth2color_subscriber import AlignedDepth2ColorSubscriber
from stretch_package.stretch_images.rgb_image_subscriber import RGBImageSubscriber
from stretch_package.stretch_movement.move_body import BaseController
from stretch_package.stretch_movement.move_head import HeadJointController
from stretch_package.stretch_movement.move_to_pose import JointPoseController
from stretch_package.stretch_movement.move_to_position import JointPositionController
from stretch_package.stretch_movement.stow_arm import StowArmController
from stretch_package.stretch_state.frame_transformer import FrameTransformer
from utils.coordinates import Pose3D
from utils.recursive_config import Config
from utils.robot_utils.advanced_movement import *
from utils.robot_utils.basic_movement import *
from utils.robot_utils.basic_perception import *
from utils.zero_shot_object_detection import detect_handle

# Tunables
VIS_BLOCK = False
SAVE_BLOCK = True

# Config & Paths (use absolute paths to avoid CWD issues)
config = Config()
scan_path = os.path.abspath(config.get_subpath("ipad_scans"))
graph_path = os.path.abspath(config.get_subpath("scene_graph"))
pcd_path  = os.path.abspath(config.get_subpath("aligned_point_clouds"))
ending    = config["pre_scanned_graphs"]["high_res"]
GRAPH_DIR = os.path.abspath(os.path.join(graph_path, ending))

def _p(path):  # pretty print existence
    ok = os.path.exists(path)
    print(f"[PATH] {'OK ' if ok else 'MISS'}: {path}")
    return ok

class TransformManager:
    def __init__(self, node: Node):
        self.node = node
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=100))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self._spin_thread = Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
        self.node.get_logger().info("TransformListener initialized.")
    def _spin(self): self.executor.spin()
    def _shutdown(self):
        self.executor.shutdown()
        self._spin_thread.join()
        self.node.get_logger().info("TransformListener stopped.")

def execute_open(drawer_id: int) -> bool:
    """
    Open a drawer by numeric ID. Returns True on success.
    """
    print("=" * 80)
    print(f"[DRAWER] Requested drawer_id: {drawer_id}")
    print(f"[DRAWER] GRAPH_DIR: {GRAPH_DIR}")
    print("=" * 80, flush=True)

    # Required files
    graph_json  = os.path.join(GRAPH_DIR, "graph.json")
    scene_json  = os.path.join(GRAPH_DIR, "scene.json")
    drawer_json = os.path.join(GRAPH_DIR, "drawers", f"{drawer_id}.json")

    missing = [p for p in [GRAPH_DIR, graph_json, scene_json, os.path.dirname(drawer_json), drawer_json] if not _p(p)]
    if missing:
        print("\n[DRAWER] ❌ Missing required path(s) above. Fix config/mounts and re-run.\n")
        return False
    
    rclpy.init(args=None)
    success = False

    # ROS nodes
    node = rclpy.create_node("transform_manager_node")
    transform_manager = TransformManager(node)
    stow_node = StowArmController()
    base_node = BaseController()
    joint_pose_node = JointPoseController()
    head_node = HeadJointController(transform_manager.tf_buffer)
    joint_position_node = JointPositionController(transform_manager.tf_buffer)
    transform_node = FrameTransformer(transform_manager.tf_buffer)

    try:
        with open(graph_json, "r") as f:
            graph_data = json.load(f)
        with open(scene_json, "r") as f:
            scene_data = json.load(f)
        with open(drawer_json, "r") as f:
            drawer_data = json.load(f)

        connections = graph_data["connections"]
        drawer_center = Pose3D(np.array(drawer_data["centroid"]))
        furniture_id = str(connections[str(drawer_id)])
        furniture_center = scene_data["furniture"][furniture_id]["centroid"]
        furniture_name = scene_data["furniture"][furniture_id]["label"]

        print(f"[DRAWER] Furniture {furniture_id} → {furniture_name}")
        print(f"[DRAWER] Drawer centroid: {drawer_center.coordinates}")

        # Coarse approach
        body_pose, front_normal = searchnet_planning.plan_drawer_search(
            furniture_name, furniture_center, drawer_center, 0.8
        )
        move_in_front_of(
            stow_node, base_node, head_node, joint_pose_node,
            body_pose, drawer_center, 0.0, 0.0, 0.0, 0.09, stow=True, grasp=True
        )

        # Detect handle using gripper camera
        rgb_img = get_rgb_picture(
            RGBImageSubscriber, joint_pose_node,
            "/gripper_camera/color/image_rect_raw", gripper=True
        )
        depth_img = get_depth_picture(
            AlignedDepth2ColorSubscriber, joint_pose_node,
            "/gripper_camera/aligned_depth_to_color/image_raw", gripper=True
        )
        handle_pose, drawer_type, _ = detect_handle(transform_node, depth_img, rgb_img)
        print("[DRAWER] HANDLE:", handle_pose)
        print("[DRAWER] TYPE  :", drawer_type)

        handle_pose.set_rot_from_direction(-front_normal)

        # Fine approach to handle
        body_pose, front_normal = searchnet_planning.plan_drawer_search(
            furniture_name, furniture_center, handle_pose, 0.6
        )

        if drawer_type == "front":
            move_in_front_of(
                stow_node, base_node, head_node, joint_pose_node,
                body_pose, handle_pose, 0.0, 0.0, np.pi/2, 0.09, stow=False, grasp=True
            )
            pull_drawer(joint_pose_node)
            look_into_drawer(joint_pose_node, handle_pose)
            get_rgb_picture(
                RGBImageSubscriber, joint_pose_node,
                "/gripper_camera/color/image_rect_raw", gripper=True,
                save_block=SAVE_BLOCK, vis_block=VIS_BLOCK
            )
            # Slight close-back (optional)
            push(joint_pose_node, handle_pose.coordinates[2] - 0.09)
        else:
            print(f"[DRAWER] Unsupported drawer_type: {drawer_type}")

        success = True

    except Exception as e:
        print(f"[DRAWER] Error: {e}")
        traceback.print_exc()
    finally:
        try:
            transform_manager._shutdown()
            node.destroy_node()
            stow_node.destroy_node()
            base_node.destroy_node()
            joint_pose_node.destroy_node()
            head_node.destroy_node()
            joint_position_node.destroy_node()
            transform_node.destroy_node()
        except Exception as e:
            print(f"[DRAWER] Cleanup error: {e}")
        rclpy.shutdown()

    return success

def _parse_args() -> int:
    p = argparse.ArgumentParser(description="Open a drawer by ID.")
    p.add_argument("--drawer_id", type=int, required=True, help="Numeric drawer id (e.g., 20, 19, 23, 16).")
    return p.parse_args().drawer_id

if __name__ == "__main__":
    did = _parse_args()
    execute_open(did)
