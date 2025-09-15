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
from utils.openmask_interface import get_mask_points
from utils.openmask_interface import get_text_similarity
from utils.open_vocab_graph_search import OpenVocabSearch
from utils.llm_utils import openai_client

# Adaptable
VIS_BLOCK = False
SAVE_BLOCK = True
NO_PROPOSALS = 3
OBJECT = "blue bottle"

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
object_location_json_path = os.path.join(config.get_subpath("scene_graph"), ending, "locations")
room_json_path = os.path.join(config.get_subpath("scene_graph"), ending, "rooms.json")
likely_furniture_label = ""
location_proposals_already_exist = False

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
    candidate_found_with_high_probability = False
    location_proposals_already_exist = False
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
            #scene_data = json.load(GRAPH_DIR + "/scene.json")
            #furniture_data = scene_data.get("furniture", {})
        
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
            
        else:     
            # Check if location proposals already exist for the object
            object_filename = f"{OBJECT.replace(' ', '_')}.json"
            object_location_file = os.path.join(object_location_json_path, object_filename)
            if os.path.exists(object_location_file):
                try:
                    with open(object_location_file, "r") as loc_file:
                        loc_data = json.load(loc_file)
                        if isinstance(loc_data, dict) and "locations" in loc_data:
                            if isinstance(loc_data["locations"], list) and len(loc_data["locations"]) == 3:
                                location_proposals_already_exist = True
                except Exception as e:
                    print(f"Error reading location proposals file: {e}")
            if location_proposals_already_exist:
                print(f"Location proposals for {OBJECT} already exist. Skipping LLM location proposal step.")
            
            else:
                candidate_found_with_high_probability = False
                print(f"{OBJECT} is NOT in the current scene graph. Switching to open vocabulary grasp search with clip and openmask3d")
                # ovs = OpenVocabSearch()
                print("Skipping open vocabulary search for now.")
                
                # candidate_found, likely_furniture_id, likely_furniture_label, mask_sim, text_sim = ovs.search(OBJECT, no_proposals=NO_PROPOSALS)
                
                # if candidate_found:
                #     probability = ovs.compute_probability(mask_sim, text_sim, 0.4)
                    
                # if candidate_found and probability > 0.4:
                #     print(f"Searching for {OBJECT} in/on {likely_furniture_label} (id: {likely_furniture_id}) with combined probability {probability:.3f}.")
                #     result = ovs.result_to_json(OBJECT, likely_furniture_id, likely_furniture_label, probability, "unknown", object_location_json_path)
                #     target_pos, furniture, front_normal, body_pose, furniture_id  = searchnet_planning.plan_furniture_search(OBJECT, 0)
                #     checked_furniture_ids.append(furniture_id)
                #     print(f"{OBJECT} is in the scene graph. Searching for it in/on {furniture} at {target_pos}")
                #     candidate_found_with_high_probability = True

                #     move_in_front_of(stow_node, base_node, head_node, joint_pose_node, body_pose, target_pos, 0.0, 0.0, 0.0, 0.0, stow=True, grasp=False)
                #     get_rgb_picture(RGBImageSubscriber, joint_pose_node, '/camera/color/image_raw', gripper=False, save_block=SAVE_BLOCK, vis_block=VIS_BLOCK)
                #     get_depth_picture(AlignedDepth2ColorSubscriber, joint_pose_node, '/camera/aligned_depth_to_color/image_raw', gripper=False, save_block=SAVE_BLOCK, vis_block=VIS_BLOCK)
                #     detected, detection_dict = yolo_detect_object(OBJECT, "head", save_block=SAVE_BLOCK)
                
               
        
        # B: Object is not in the scene graph    
        if detected == False and location_proposals_already_exist == False:          
            # Check for object at the different locations proposed by DeepSeek
            print(f"No suitable furniture found for {OBJECT} with high enough probability. Asking openai for likely locations.")
            oai = openai_client.oai_client
            result = openai_client.ask_for_shelf_with_room_json(oai, room_json_path, OBJECT, "end table near door", "gpt-4o-mini")

            filename = f"{OBJECT.replace(' ', '_')}.json"
            object_location_json_path_llm_filename = os.path.join(object_location_json_path, filename)
            print(f"Saving object location prediction to {object_location_json_path_llm_filename}")
            with open(object_location_json_path_llm_filename, 'w') as f:
                json.dump(result, f, indent=4)
            location_proposals_already_exist = True
            
        if (not candidate_found_with_high_probability or not detected) and location_proposals_already_exist:
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
                    print(f"Found {OBJECT} in/on {furniture}: {detection_dict}")
                    break
                else:
                    print(f"Did not find {OBJECT} in/on {furniture}.")
        
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
    # cd workspace/stretch-compose/source/gpd
    # ./run_docker_new.sh
    execute_search(OBJECT)