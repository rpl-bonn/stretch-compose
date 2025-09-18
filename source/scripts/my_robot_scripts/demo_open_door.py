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
from utils.zero_shot_object_detection import yolo_detect_object, detect_handle, detect_door_handle

# Adaptable
VIS_BLOCK = False
SAVE_BLOCK = True
NO_PROPOSALS = 3

# Config and Paths
config = Config()
scan_path = config.get_subpath("ipad_scans")
graph_path = config.get_subpath("scene_graph")
pcd_path = config.get_subpath("aligned_point_clouds")
ending = config["pre_scanned_graphs"]["high_res"]
SCAN_DIR = os.path.join(scan_path, ending)
GRAPH_DIR = os.path.join(graph_path, ending)
PCD_DIR = os.path.join(pcd_path, ending)
IMG_DIR = config.get_subpath("images")


class TransformManager:
    def __init__(self, node: Node):
        self.node = node
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=100))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)
        self.node.get_logger().info("TransformListener initialized.")
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self._spin_thread = Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
        
    def _spin(self):
        self.executor.spin()
            
    def _shutdown(self):
        self.executor.shutdown()
        self._spin_thread.join()
        self.node.get_logger().info("TransformListener stopped.")

    
def execute_search(drawer_id: int) -> bool:
    """
    Search for a queried object inside the drawers of the environment.

    Args:
        OBJECT (str): Object to search for and grasp.

    Returns:
        bool: True if the search and grasp was successful, False otherwise.
    """
    rclpy.init(args=None)
    success = False
    
    # Initialize ROS nodes
    node = rclpy.create_node('transform_manager_node')
    transform_manager = TransformManager(node)
    stow_node = StowArmController()
    base_node = BaseController()
    joint_pose_node = JointPoseController()
    head_node = HeadJointController(transform_manager.tf_buffer)
    joint_position_node = JointPositionController(transform_manager.tf_buffer)
    transform_node = FrameTransformer(transform_manager.tf_buffer)
    
    try: 
        print(f"Opening drawer {drawer_id}.")
        with open(os.path.join(GRAPH_DIR, "graph.json"), "r") as file:
            graph_data = json.load(file)
        with open(os.path.join(GRAPH_DIR, "scene.json"), "r") as file:
            scene_data = json.load(file)
        connections = graph_data["connections"]
        
        ## Check each drawer for the object    
        print("-----------------############################-----------------")

        with open(os.path.join(GRAPH_DIR, "drawers", f"{drawer_id}.json"), "r") as file:
            drawer_data = json.load(file)
        drawer_center = Pose3D(np.array(drawer_data["centroid"]))
        furniture_id = str(connections[str(drawer_id)])
        furniture_center = scene_data["furniture"][furniture_id]["centroid"]
        furniture_name = scene_data["furniture"][furniture_id]["label"]
            
        # Move robot in front of the drawer
        body_pose, front_normal = searchnet_planning.plan_drawer_search(furniture_name, furniture_center, drawer_center, 0.8)
            
        move_in_front_of(stow_node, base_node, head_node, joint_pose_node, body_pose, drawer_center, 0.0, 0.0, 0.0, 0.09, stow=True, grasp=True)           
            
        # Take image of drawer to detect handle
        rgb_img = get_rgb_picture(RGBImageSubscriber, joint_pose_node, "/gripper_camera/color/image_rect_raw", gripper=True)
        depth_img = get_depth_picture(AlignedDepth2ColorSubscriber, joint_pose_node, "/gripper_camera/aligned_depth_to_color/image_raw", gripper=True)
        # Skip if no handle detected
        
        handle_pose, drawer_type, _ = detect_door_handle(transform_node, depth_img, rgb_img)
        print("----------------------------------------------------------")
        print(f"HANDLE POSE: {handle_pose}")
        print("----------------------------------------------------------")
        print(f"DRAWER TYPE: {drawer_type}")
        print("----------------------------------------------------------")
        
        
        handle_pose.set_rot_from_direction(-front_normal)
        print("Next check Plan Drawer Search towards handle")
        # Refine robot position towards handle
        body_pose, front_normal = searchnet_planning.plan_door_search(furniture_name, furniture_center, handle_pose, 0.5, rgb_img)
        print(f'Body pose: {body_pose.as_ndarray()}')
        # Open drawer and check for object inside
        if drawer_type == "front":
            move_in_front_of(stow_node, base_node, head_node, joint_pose_node, body_pose, handle_pose, 0.0, 0.0, np.pi/2, 0.09, stow=False, grasp=True)
            pull_door(joint_pose_node)
            look_into_drawer(joint_pose_node, handle_pose)
            get_rgb_picture(RGBImageSubscriber, joint_pose_node, "/gripper_camera/color/image_rect_raw", gripper=True, save_block=SAVE_BLOCK, vis_block=VIS_BLOCK)
            # detected, detection_dict = yolo_detect_object(OBJECT, "gripper", conf=0.2, save_block=SAVE_BLOCK)
            push(joint_pose_node, handle_pose.coordinates[2]-0.09)
            
        if drawer_type == "right":
            move_in_front_of(stow_node, base_node, head_node, joint_pose_node, body_pose, handle_pose, 0.0, 0.0, np.pi/2, 0.09, stow=False, grasp=True)
            pull_door(joint_pose_node)
            # look_into_drawer(joint_pose_node, handle_pose)
            # get_rgb_picture(RGBImageSubscriber, joint_pose_node, "/gripper_camera/color/image_rect_raw", gripper=True, save_block=SAVE_BLOCK, vis_block=VIS_BLOCK)
            # # detected, detection_dict = yolo_detect_object(OBJECT, "gripper", conf=0.2, save_block=SAVE_BLOCK)
            # push(joint_pose_node, handle_pose.coordinates[2]-0.09)
        
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        success = False
    
    finally:               
        # Destroy nodes
        transform_manager._shutdown()
        node.destroy_node()  
        stow_node.destroy_node()
        base_node.destroy_node()
        joint_pose_node.destroy_node()
        head_node.destroy_node()
        joint_position_node.destroy_node()
        transform_node.destroy_node()
        rclpy.shutdown()

    return success
  
     
if __name__ == "__main__":
    # docker run -p 5004:5004 --gpus all -it craiden/yolodrawer:v1.0 python3 app.py
    # args = os.sys.argv[2]
    id=21
    execute_search(id)