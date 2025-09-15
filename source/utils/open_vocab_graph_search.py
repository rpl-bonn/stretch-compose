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
from utils.openmask_interface import get_mask_points, get_text_similarity

# Adaptable
VIS_BLOCK = False
SAVE_BLOCK = True
NO_PROPOSALS = 3
OBJECT = "sugar box"

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


class OpenVocabSearch:
    def __init__(self):
        self.pcd = o3d.io.read_point_cloud(str(os.path.join(PCD_DIR, "scene.ply")))

        try:
            with open(os.path.join(GRAPH_DIR, "graph.json"), "r") as file:
                self.graph_data = json.load(file)
                
        except: 
            print("Failed to load scene graph data.")
            traceback.print_exc()
            return
        
        try:
            with open(os.path.join(GRAPH_DIR, "scene.json"), "r") as file:
                furniture_json = json.load(file)
                self.furniture_data = furniture_json.get("furniture", {})
        except:
            print("Failed to load furniture data.")
            traceback.print_exc()
            return
    def search(self, object_name, no_proposals=3):
        pcd_obj, pcd_env, mask_sim = get_mask_points(OBJECT, config, idx=0, vis_block=VIS_BLOCK)
        
        if mask_sim < 0.25:
            print(f"Low mask similarity {mask_sim:.3f}. The object '{OBJECT}' may not be well represented in the point cloud.")
            return False, None, None, mask_sim, 0.0
            
        # Compute centroid and bounding box of pcd_obj
        obj_points = np.asarray(pcd_obj.points)
        obj_centroid = obj_points.mean(axis=0)
        obj_min = obj_points.min(axis=0)
        obj_max = obj_points.max(axis=0)
        obj_bbox = (obj_min, obj_max)

        print(f"Object {OBJECT}  centroid: {obj_centroid}, bounding box: {obj_bbox}")
        

        # Find which furniture the object most likely belongs to
        likely_furniture_id = None
        likely_furniture_label = None
        checked_furniture_ids = []
        min_dist = float('inf')
        for fid, furn in self.furniture_data.items():
            furn_centroid = np.array(furn["centroid"])
            furn_dims = np.array(furn["dimensions"])
            print(f"Furniture {fid}, {furn['label']} centroid: {furn_centroid}, dimensions: {furn_dims}")
            # Compute furniture bbox
            furn_min = furn_centroid - furn_dims / 2
            furn_max = furn_centroid + furn_dims / 2
            # Check if object centroid is inside furniture bbox
            
            # Check if object centroid is inside furniture bbox or within 0.5m of the top plane
            margin = 0.2  # 5cm margin
            inside = np.all(obj_centroid >= (furn_min - margin)) and np.all(obj_centroid <= (furn_max + margin))
            print(f"Object centroid: {obj_centroid}, Furniture '{furn['label']}' bbox min: {furn_min}, max: {furn_max}. Centroid is {'inside' if inside else 'NOT inside'} the furniture bounding box.")
            
            # Top plane z value
            top_plane_z = furn_max[2]
            print(f"top plane z: {top_plane_z}")
            # Within 0.5m above the top plane and horizontally within bbox
            # Define the top plane of the furniture (z = top_plane_z)
            # Create a cuboid region above the top plane with margin in x, y, and z
            xy_margin = 0.10  # 10cm margin in x and y
            z_margin_below = 0.10  # No margin below the top plane
            z_margin_above = 0.10  # 10cm above the top plane

            # Object's base z (lowest z of its bbox)
            obj_base_z = obj_min[2]

            # Check if object's base is within the top plane cuboid (with margin)
            near_top_plane = (
                (obj_base_z >= top_plane_z - z_margin_below) and
                (obj_base_z <= top_plane_z + z_margin_above) and
                (obj_centroid[0] >= furn_min[0] - xy_margin) and
                (obj_centroid[0] <= furn_max[0] + xy_margin) and
                (obj_centroid[1] >= furn_min[1] - xy_margin) and
                (obj_centroid[1] <= furn_max[1] + xy_margin)
            )
            
            print(f"Furniture '{furn['label']}': Top plane z = {top_plane_z}, Object centroid z = {obj_centroid[2]}, centroid is {'near' if near_top_plane else 'not near'} the top plane.")
            inside_or_on_top = inside or near_top_plane
            # Compute distance from object centroid to furniture centroid
            dist = np.linalg.norm(obj_centroid - furn_centroid)
            if inside_or_on_top and dist < min_dist:
                likely_furniture_id = fid
                likely_furniture_label = furn["label"]
                min_dist = dist

        if likely_furniture_id is not None:
            print(f"Object {object_name} most likely belongs to furniture: {likely_furniture_label} (id: {likely_furniture_id})")
            text_sim = get_text_similarity(object_name, likely_furniture_label)
            print(f"Similarity between '{object_name}' and '{likely_furniture_label}': {text_sim:.3f}")
            checked_furniture_ids.append(likely_furniture_id)
            return True, likely_furniture_id, likely_furniture_label, mask_sim, text_sim
        else:
            print(f"Could not confidently assign {object_name} to any furniture.")
            return False, None, None, mask_sim, 0.0

    def compute_probability(self, mask_sim: float, text_sim: float, alpha: float=0.3) -> float:
        return alpha * mask_sim + (1 - alpha) * text_sim

    def result_to_json(self, object_name, fid, flabel, prob, room_name, object_location_json_path) -> dict:
        result = {
        "item": object_name,
        "locations": [
            {
                "furniture_id": fid,
                "furniture_name": flabel,
                "relation": "on top of",  # dummy default
                "probability": round(prob, 3),
                "room": room_name.lower()
            }
        ]
        }
        os.makedirs(object_location_json_path, exist_ok=True)
        # file name: object_name.json (spaces -> underscores)
        filename = f"{object_name.replace(' ', '_')}.json"
        filepath = os.path.join(object_location_json_path, filename)

        with open(filepath, "w") as f:
            json.dump(result, f, indent=4)

        print(f"Saved JSON for '{object_name}' to {filepath}")
        return result

if __name__ == "__main__":
    ovs = OpenVocabSearch()
    candidate_found, likely_furniture_id, likely_furniture_label, mask_sim, text_sim = ovs.search(OBJECT, no_proposals=NO_PROPOSALS)
    object_location_json_path = os.path.join(config.get_subpath("scene_graph"), ending, "locations")
        
    if candidate_found:
        probability = ovs.compute_probability(mask_sim, text_sim, 0.4)
        
    if candidate_found and probability > 0.4:
        print(f"Searching for {OBJECT} in/on {likely_furniture_label} (id: {likely_furniture_id}) with combined probability {probability:.3f}.")
        result = ovs.result_to_json(OBJECT, likely_furniture_id, likely_furniture_label, probability, "unknown", object_location_json_path)
