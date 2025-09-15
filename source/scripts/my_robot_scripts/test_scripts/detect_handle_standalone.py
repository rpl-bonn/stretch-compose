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

def get_handle_pose():
    rclpy.init(args=None)
    # Initialize ROS nodes
    node = rclpy.create_node('home_pose_node')
    transform_manager = TransformManager(node)
    stow_node = StowArmController()
    base_node = BaseController()
    joint_pose_node = JointPoseController()
    head_node = HeadJointController(transform_manager.tf_buffer)
    joint_position_node = JointPositionController(transform_manager.tf_buffer)
    transform_node = FrameTransformer(transform_manager.tf_buffer)


    # Adaptable
    VIS_BLOCK = False
    SAVE_BLOCK = True
    NO_PROPOSALS = 3
    OBJECT = "tennis ball"

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

    print("All ROS nodes initialized.")

    try:
        print(f"Searching for {OBJECT} in the environment...")
        with open(os.path.join(GRAPH_DIR, "graph.json"), "r") as file:
            graph_data = json.load(file)
        with open(os.path.join(GRAPH_DIR, "scene.json"), "r") as file:
            scene_data = json.load(file)
        connections = graph_data["connections"]

        rgb_img = get_rgb_picture(RGBImageSubscriber, joint_pose_node, "/gripper_camera/color/image_rect_raw", gripper=True)
        depth_img = get_depth_picture(AlignedDepth2ColorSubscriber, joint_pose_node, "/gripper_camera/aligned_depth_to_color/image_raw", gripper=True)

        handle_pose, drawer_type, hinge_pose = detect_door_handle(transform_node, depth_img, rgb_img)
        print("----------------------------------------------------------")
        print(f"HANDLE POSE: {handle_pose}")
        print("----------------------------------------------------------")
        print(f"DRAWER TYPE: {drawer_type}")
        print("----------------------------------------------------------")
        print(f"HINGE POSE: {hinge_pose}")
        print("----------------------------------------------------------")
        
    except Exception as e:
        print("An error occurred during the operation:")
        print(str(e))
        traceback.print_exc()
        
    finally:
        transform_manager._shutdown()
        node.destroy_node()
        rclpy.shutdown()

    
def main(args=None):
    get_handle_pose()
    
if __name__ == "__main__":
    main()