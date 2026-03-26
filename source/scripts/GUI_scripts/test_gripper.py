#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import numpy as np
import open3d as o3d


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

from utils.robot_utils.global_parameters import *
from utils.robot_utils.basic_perception import check_object_distance
from utils.robot_utils.basic_movement import *
from scipy.spatial.transform import Rotation
from utils import vis
from utils.coordinates import Pose2D, Pose3D, from_a_to_b_distanced, pose_distanced, get_door_opening_poses
from utils.graspnet_interface import predict_full_grasp
from gpd.gpd_client_api import predict_full_grasp as gpd_predict_full_grasp
from utils.importer import PointCloud
from utils.point_clouds import icp
from utils.recursive_config import Config
from utils.time import convert_time
from scripts.my_robot_scripts.graspnet_testing import visualize_grasps
from stretch_package.stretch_movement.move_body import BaseController
from stretch_package.stretch_movement.move_to_pose import JointPoseController
from stretch_package.stretch_movement.move_to_position import JointPositionController
from stretch_package.stretch_movement.stow_arm import StowArmController
from stretch_package.stretch_state.frame_transformer import FrameTransformer
import os
from utils.robot_utils.advanced_movement import *

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
        
def open_gripper():
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
    # set_gripper(joint_pose_node, -0.6)
    open_door(joint_pose_node)
    
if __name__ == "__main__":
    try:
        open_gripper()
    except Exception as e:
        print(f"[GRASP] ERROR: {e}")
        traceback.print_exc()
    finally:
        print("[GRASP] Shutting down...")
        rclpy.shutdown()
        print("[GRASP] Shutdown complete.")