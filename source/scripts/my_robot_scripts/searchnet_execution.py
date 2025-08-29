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

from scripts.llm_scripts.deepseek_client import ask_deepseek_for_locations
from scripts.my_robot_scripts import searchnet_planning
from scripts.preprocessing_scripts.scenegraph_preprocessing import add_object_to_scene_graph, update_object_in_scene_graph
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
from utils.time import convert_time
from utils.zero_shot_object_detection import yolo_detect_object, detect_handle

# Adaptable
VIS_BLOCK = False
SAVE_BLOCK = True
DEEPSEEK = False
NO_PROPOSALS = 3
OBJECT = "bottle"
HINT = ""

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
    Execute the search and grasping of an object using the robot.
    This function initializes the necessary nodes, performs the search using DeepSeek and SearchNet,
    and attempts to grasp the object if found.

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
        with open(os.path.join(GRAPH_DIR, "scene.json"), "r") as file:
            scene_data = json.load(file)
        
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
            # Ask deepseek for most likely object location
            if not os.path.exists(os.path.join(GRAPH_DIR, "locations", f"{OBJECT.replace(' ', '_')}.json")) or DEEPSEEK:
                print(f"{OBJECT} not found in scene graph. Asking DeepSeek for locations...")
                ask_deepseek_for_locations(OBJECT, HINT, NO_PROPOSALS)
                
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
        
        # C: Object is in open space
        if detected:
            print(f"Found {OBJECT} in/on {furniture}: {detection_dict}")
            center, body_pose, pcd = searchnet_planning.plan_object_search(transform_node, detection_dict, front_normal, pcd, furniture_id)
            move_in_front_of(stow_node, base_node, head_node, joint_pose_node, body_pose, center, 0.0, 0.0, 0.0, 0.1, stow=False, grasp=True)
            
            # Get dynamic point cloud of object
            gripper_tform_map = transform_node.get_tf_matrix("map", "link_grasp_center")
            start_pose = np.array((0.0, 0.0, 0.0, 1.0)) @ gripper_tform_map.T
            start_pose = Pose3D(start_pose[:3], center.rot_matrix.copy())
            pcd_obj, pcd_env = collect_dynamic_point_cloud(OBJECT, joint_position_node, joint_pose_node, transform_node, start_pose, center, pcd)
            
            # Add/Update object to/in the scene graph
            if not in_scene_graph:    
                obj_id = graph_data["node_ids"][-1] + 1
                add_object_to_scene_graph(OBJECT, obj_id, center.as_ndarray(), detection_dict["confidence"].item(), pcd_obj, furniture_id, -1)
            else:
                obj_id = graph_data["node_ids"][graph_data["node_labels"].index(OBJECT)]
                update_object_in_scene_graph(OBJECT, obj_id, center.as_ndarray(), detection_dict["confidence"].item(), pcd_obj, furniture_id, -1)
                
            # Find grasp position
            find_new_grasp_dynamically(joint_position_node, joint_pose_node, transform_node, body_pose, 0.15, 0.05, config, pcd_obj, pcd_env)
            success = True
        
        # D: Object is not in open spaces
        if not detected:
            #checked_furniture_ids = ["8", "16", "9"]
            print(f"Object {OBJECT} not found in open spaces. Searching in concealed spaces...")
            connections = graph_data["connections"]
            
            # Filter drawers with semantic and geometric reasoning
            fitting_drawers = searchnet_planning.filter_drawers(in_scene_graph, graph_data, connections, checked_furniture_ids, OBJECT)
            
            # Check each drawer for the object    
            for drawer_id in fitting_drawers:
                with open(os.path.join(GRAPH_DIR, "drawers", f"{drawer_id}.json"), "r") as file:
                    drawer_data = json.load(file)
                drawer_center = Pose3D(np.array(drawer_data["centroid"]))
                furniture_id = str(connections[str(drawer_id)])
                furniture_center = scene_data["furniture"][furniture_id]["centroid"]
                furniture_name = scene_data["furniture"][furniture_id]["label"]
                
                # Move in front of the drawer
                body_pose, front_normal = searchnet_planning.plan_drawer_search(furniture_name, furniture_center, drawer_center, 0.8)
                move_in_front_of(stow_node, base_node, head_node, joint_pose_node, body_pose, drawer_center, 0.0, 0.0, 0.0, 0.09, stow=True, grasp=True)           
                
                # Take image of drawer to detect handle
                rgb_img = get_rgb_picture(RGBImageSubscriber, joint_pose_node, "/gripper_camera/color/image_rect_raw", gripper=True)
                depth_img = get_depth_picture(AlignedDepth2ColorSubscriber, joint_pose_node, "/gripper_camera/aligned_depth_to_color/image_raw", gripper=True)
                try:
                    handle_pose, drawer_type, hinge_pose = detect_handle(transform_node, depth_img, rgb_img)
                except Exception as e:
                    continue
                print(f"OPENING DIRECTION: {drawer_type}")
                handle_pose.set_rot_from_direction(-front_normal)
                
                # Refine position towards handle
                body_pose, front_normal = searchnet_planning.plan_drawer_search(furniture_name, furniture_center, handle_pose, 0.6)
            
                # Open drawer and check for object inside
                if drawer_type == "front":
                    move_in_front_of(stow_node, base_node, head_node, joint_pose_node, body_pose, handle_pose, 0.0, 0.0, np.pi/2, 0.09, stow=False, grasp=True)
                    pull_drawer(joint_pose_node)
                    look_into_drawer(joint_pose_node, handle_pose)
                    get_rgb_picture(RGBImageSubscriber, joint_pose_node, "/gripper_camera/color/image_rect_raw", gripper=True, save_block=SAVE_BLOCK, vis_block=VIS_BLOCK)
                    detected, detection_dict = yolo_detect_object(OBJECT, "gripper", conf=0.2, save_block=SAVE_BLOCK)
                    push(joint_pose_node, handle_pose.coordinates[2]-0.09)
                # Check door for object behind                   
                elif drawer_type == "left" or drawer_type == "right":
                    time.sleep(3.0) # Wait for manual drawer opening
                    set_gripper(joint_pose_node, True)
                    get_rgb_picture(RGBImageSubscriber, joint_pose_node, "/gripper_camera/color/image_rect_raw", gripper=True, save_block=SAVE_BLOCK, vis_block=VIS_BLOCK)
                    detected, detection_dict = yolo_detect_object(OBJECT, "gripper", conf=0.2, save_block=SAVE_BLOCK)
                
                # Add/Update object to/in the scene graph
                if detected:
                    print(f"Found {OBJECT} in door {drawer_id} in {furniture_name}.")
                    if not in_scene_graph:    
                        obj_id = graph_data["node_ids"][-1] + 1
                        add_object_to_scene_graph(OBJECT, obj_id, drawer_center.as_ndarray(), detection_dict["confidence"].item(), None, furniture_id, drawer_id)
                    else:
                        obj_id = graph_data["node_ids"][graph_data["node_labels"].index(OBJECT)]
                        update_object_in_scene_graph(OBJECT, obj_id, drawer_center.as_ndarray(), detection_dict["confidence"].item(), None, furniture_id, drawer_id)
                    success = True
                    break
    
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
    # docker run --net=bridge --mac-address=02:42:ac:11:00:02 -p 5000:5000 --gpus all -it craiden/graspnet:v1.0 python3 app.py
    # docker run -p 5004:5004 --gpus all -it craiden/yolodrawer:v1.0 python3 app.py
    # ./run_docker_new.sh
    execute_search(OBJECT)