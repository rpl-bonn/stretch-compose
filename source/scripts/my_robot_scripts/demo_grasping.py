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
from utils.point_clouds import collect_dynamic_point_cloud
from utils.recursive_config import Config
from utils.robot_utils.advanced_movement import *
from utils.robot_utils.basic_movement import *
from utils.robot_utils.basic_perception import *
from utils.zero_shot_object_detection import yolo_detect_object

# Adaptable
VIS_BLOCK = False
SAVE_BLOCK = True
NO_PROPOSALS = 3
OBJECT = "bottle"

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

    
def execute_search(OBJECT: str) -> bool:
    """
    Search for queried object in open space locations.

    Args:
        OBJECT (str): Object to search for and grasp.

    Returns:
        bool: True if the search and grasp was successful, False otherwise.
    """
    rclpy.init(args=None)
    success = False
    detected = False
    in_scene_graph = False
    checked_furniture_ids = []
    pcd = o3d.io.read_point_cloud(str(os.path.join(PCD_DIR, "scene.ply")))
    
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
        with open(os.path.join(GRAPH_DIR, "graph.json"), "r") as file:
            graph_data = json.load(file)
        
        # A: Object is in the scene graph
        if OBJECT in graph_data["node_labels"]:
            in_scene_graph = True
            target_pos, furniture, front_normal, body_pose, furniture_id  = searchnet_planning.plan_furniture_search(OBJECT)
            checked_furniture_ids.append(furniture_id)
            print(f"{OBJECT} is in the scene graph. Searching for it in/on {furniture} at {target_pos}")

            move_in_front_of(stow_node, base_node, head_node, joint_pose_node, body_pose, target_pos, 0.0, 0.0, 0.0, 0.0, stow=True, grasp=False)
            get_rgb_picture(RGBImageSubscriber, joint_pose_node, '/camera/color/image_raw', gripper=False, save_block=SAVE_BLOCK, vis_block=VIS_BLOCK)
            get_depth_picture(AlignedDepth2ColorSubscriber, joint_pose_node, '/camera/aligned_depth_to_color/image_raw', gripper=False, save_block=SAVE_BLOCK, vis_block=VIS_BLOCK)
            detected, detection_dict = yolo_detect_object(OBJECT, "head", save_block=SAVE_BLOCK)
        
        # B: Object is not in the scene graph    
        if not detected:           
            # Check for object at the different locations proposed by DeepSeek
            for i in range(NO_PROPOSALS):
                target_pos, furniture, front_normal, body_pose, furniture_id  = searchnet_planning.plan_furniture_search(OBJECT, i)
                # Skip if already checked
                if furniture_id in checked_furniture_ids:
                    print(f"Already checked {furniture} ({furniture_id}).")
                    continue
                checked_furniture_ids.append(furniture_id)
                print(f"Searching for {OBJECT} in/on {furniture} ({target_pos}).")
                
                move_in_front_of(stow_node, base_node, head_node, joint_pose_node, body_pose, target_pos, 0.0, 0.0, 0.0, 0.0, stow=True, grasp=False)
                get_rgb_picture(RGBImageSubscriber, joint_pose_node, '/camera/color/image_raw', gripper=False, save_block=SAVE_BLOCK, vis_block=VIS_BLOCK)
                get_depth_picture(AlignedDepth2ColorSubscriber, joint_pose_node, '/camera/aligned_depth_to_color/image_raw', gripper=False, save_block=SAVE_BLOCK, vis_block=VIS_BLOCK)
                detected, detection_dict = yolo_detect_object(OBJECT, "head", save_block=SAVE_BLOCK)
                if detected:
                    break
        
        # C: Graps object in open space
        if detected:
            print(f"Found {OBJECT} in/on {furniture}: {detection_dict}")
            # Move closer to the object
            center, body_pose, pcd = searchnet_planning.plan_object_search(transform_node, detection_dict, front_normal, pcd, furniture_id)
            move_in_front_of(stow_node, base_node, head_node, joint_pose_node, body_pose, center, 0.0, 0.0, 0.0, 0.1, stow=False, grasp=True)
            
            # Get dynamic point cloud of object
            gripper_tform_map = transform_node.get_tf_matrix("map", "link_grasp_center")
            start_pose = np.array((0.0, 0.0, 0.0, 1.0)) @ gripper_tform_map.T
            start_pose = Pose3D(start_pose[:3], center.rot_matrix.copy())
            pcd_obj, pcd_env = collect_dynamic_point_cloud(OBJECT, joint_position_node, joint_pose_node, transform_node, start_pose, center, pcd)
                
            # Find grasp position
            find_new_grasp_dynamically(joint_position_node, joint_pose_node, transform_node, body_pose, 0.15, 0.05, config, pcd_obj, pcd_env)
            success = True
    
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
    # cd workspace/stretch-compose/source/gpd && ./run_docker_new.sh
    execute_search(OBJECT)
