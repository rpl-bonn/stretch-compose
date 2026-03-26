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
import open3d as o3d  # make sure you have pip install open3d

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
from utils.point_clouds import collect_dynamic_point_cloud
from utils.recursive_config import Config
from utils.robot_utils.advanced_movement import *
from utils.robot_utils.basic_movement import *
from utils.robot_utils.basic_perception import *
from utils.zero_shot_object_detection import yolo_detect_object

# Defaults
VIS_BLOCK_DEFAULT = False
SAVE_BLOCK_DEFAULT = True
NO_PROPOSALS_DEFAULT = 3

# Config paths
config = Config()
scan_path = config.get_subpath("ipad_scans")
graph_path = config.get_subpath("scene_graph")
pcd_path = config.get_subpath("aligned_point_clouds")
ending = config["pre_scanned_graphs"]["high_res"]
SCAN_DIR = os.path.join(scan_path, ending)
GRAPH_DIR = os.path.join(graph_path, ending)
PCD_DIR = os.path.join(pcd_path, ending)

class TransformManager:
    def __init__(self, node: Node):
        self.node = node
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=100))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self._spin_thread = Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
    def _spin(self): self.executor.spin()
    def _shutdown(self): self.executor.shutdown(); self._spin_thread.join()

def execute_search(object_name: str, vis_block=False, save_block=True, no_proposals=3) -> bool:
    print("="*70)
    print(f"[GRASP] Requested object: {object_name}")
    print("="*70, flush=True)

    
    rclpy.init(args=None)
    success, detected = False, False
    checked_furniture_ids = []

    # Load scene point cloud
    scene_pcd_path = os.path.join(PCD_DIR, "scene.ply")
    if not os.path.isfile(scene_pcd_path):
        print(f"[GRASP] ERROR: Missing scene point cloud at {scene_pcd_path}")
        return False
    pcd = o3d.io.read_point_cloud(scene_pcd_path)

    node = rclpy.create_node("transform_manager_node")
    transform_manager = TransformManager(node)
    stow_node, base_node = StowArmController(), BaseController()
    joint_pose_node, head_node = JointPoseController(), HeadJointController(transform_manager.tf_buffer)
    joint_position_node = JointPositionController(transform_manager.tf_buffer)
    transform_node = FrameTransformer(transform_manager.tf_buffer)

    try:
        with open(os.path.join(GRAPH_DIR, "graph.json"), "r") as file:
            graph_data = json.load(file)

        if object_name in graph_data.get("node_labels", []):
            
            target_pos, furniture, front_normal, body_pose, fid = searchnet_planning.plan_furniture_search(object_name)
            checked_furniture_ids.append(fid)
            print(f"[GRASP] In scene graph → {furniture} at {target_pos}")
            move_in_front_of(stow_node, base_node, head_node, joint_pose_node, body_pose, target_pos,
                             0.0,0.0,0.0,0.0, stow=True, grasp=False)
            get_rgb_picture(RGBImageSubscriber, joint_pose_node, "/camera/color/image_raw", gripper=False,
                            save_block=save_block, vis_block=vis_block)
            get_depth_picture(AlignedDepth2ColorSubscriber, joint_pose_node, "/camera/aligned_depth_to_color/image_raw",
                              gripper=False, save_block=save_block, vis_block=vis_block)
            detected, detection_dict = yolo_detect_object(object_name, "head", save_block=save_block)

        if not detected:
            for i in range(no_proposals):
                target_pos, furniture, front_normal, body_pose, fid = searchnet_planning.plan_furniture_search(object_name, i)
                if fid in checked_furniture_ids: continue
                checked_furniture_ids.append(fid)
                print(f"[GRASP] Proposal {i+1}: {furniture} at {target_pos}")
                move_in_front_of(stow_node, base_node, head_node, joint_pose_node, body_pose, target_pos,
                                 0.0,0.0,0.0,0.0, stow=True, grasp=False)
                get_rgb_picture(RGBImageSubscriber, joint_pose_node, "/camera/color/image_raw", gripper=False,
                                save_block=save_block, vis_block=vis_block)
                get_depth_picture(AlignedDepth2ColorSubscriber, joint_pose_node, "/camera/aligned_depth_to_color/image_raw",
                                  gripper=False, save_block=save_block, vis_block=vis_block)
                detected, detection_dict = yolo_detect_object(object_name, "head", save_block=save_block)
                if detected: break

        if detected:
            print(f"[GRASP] Found {object_name}: {detection_dict}")
            center, body_pose, pcd = searchnet_planning.plan_object_search(transform_node, detection_dict, front_normal, pcd, fid)
            move_in_front_of(stow_node, base_node, head_node, joint_pose_node, body_pose, center,
                             0.0,0.0,0.0,0.1, stow=False, grasp=True)
            gripper_tform_map = transform_node.get_tf_matrix("map","link_grasp_center")
            start_pose = Pose3D((np.array((0,0,0,1)) @ gripper_tform_map.T)[:3], center.rot_matrix.copy())
            pcd_obj, pcd_env = collect_dynamic_point_cloud(object_name, joint_position_node, joint_pose_node,
                                                           transform_node, start_pose, center, pcd)
            find_new_grasp_dynamically(joint_position_node, joint_pose_node, transform_node, body_pose,
                                       0.15,0.05,config,pcd_obj,pcd_env)
            success = True
        else:
            print(f"[GRASP] Could not detect {object_name}")

    except Exception as e:
        print(f"[GRASP] Error: {e}"); traceback.print_exc()
    finally:
        transform_manager._shutdown()
        for n in [node, stow_node, base_node, joint_pose_node, head_node, joint_position_node, transform_node]:
            try: n.destroy_node()
            except: pass
        rclpy.shutdown()

    return success

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--object","-o", required=True, help="Object name (can have spaces)")
    p.add_argument("--vis_block", action="store_true", default=VIS_BLOCK_DEFAULT)
    p.add_argument("--save_block", action="store_true", default=SAVE_BLOCK_DEFAULT)
    p.add_argument("--proposals", type=int, default=NO_PROPOSALS_DEFAULT)
    return p.parse_args()

if __name__=="__main__":
    args = _parse_args()
    print(f"------------------{args.object}------------------")
    execute_search(args.object, vis_block=args.vis_block, save_block=args.save_block, no_proposals=args.proposals)
