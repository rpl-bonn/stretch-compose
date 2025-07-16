import numpy as np
from utils.recursive_config import Config
import os
from collections import OrderedDict
import re
import json
import open3d as o3d
from gpd.gpd_client_api import predict_full_grasp as gpd_predict_full_grasp
import rclpy
from rclpy.node import Node
from stretch_package.stretch_movement.stow_arm import StowArmController
from stretch_package.stretch_movement.move_body import BaseController
from stretch_package.stretch_movement.move_to_pose import JointPoseController
from stretch_package.stretch_state.frame_transformer import FrameTransformer
from scripts.my_robot_scripts.searchnet_execution import TransformManager
from utils.robot_utils.basic_perception import check_object_distance
from scripts.my_robot_scripts.searchnet_planning import get_mask_points, get_shelf_front_normal
from utils.robot_utils.advanced_movement import drive_home


def check_furniture_visibility(camera_json, furniture_json, icp_tform_ground, image_width, image_height):
    visible_furniture = []

    # Parse camera data
    intrinsics = np.array(camera_json["intrinsics"]).reshape(3, 3)
    camera_pose = np.array(camera_json["cameraPoseARFrame"]).reshape(4, 4)
    scan_tform = np.array(icp_tform_ground).reshape(4, 4)

    for _, furniture in furniture_json["nodes"].items():
        label = furniture["label"]
        world_coords = np.array(furniture["centroid"])
        dimensions = np.array(furniture["dimensions"])

        # Transform furniture position to camera space
        scan_coords = np.linalg.inv(scan_tform) @ np.append(world_coords, 1)
        camera_coords = np.linalg.inv(camera_pose) @ scan_coords
        
        # Project to image plane
        #if camera_coords[2] >= 0:
            #continue
        #u = (intrinsics[0, 0] * camera_coords[0] / camera_coords[2]) + intrinsics[0, 2]
        #v = (intrinsics[1, 1] * camera_coords[1] / camera_coords[2]) + intrinsics[1, 2]  
        #if 0 <= u < image_width and 0 <= v < image_height:
        #    visible_furniture.append(label)      

        offsets = np.array([
            [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5], [-0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5], [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5], [0.5, 0.5, -0.5]
        ]) * dimensions

        object_corners = [world_coords + offset for offset in offsets]

        # Project all corners to the image plane
        projected_corners = []
        for corner in object_corners:
            corner_scan = np.linalg.inv(scan_tform) @ np.append(corner, 1)
            corner_camera = np.linalg.inv(camera_pose) @ corner_scan

            if corner_camera[2] < 0:
                u = (intrinsics[0, 0] * corner_camera[0] / corner_camera[2]) + intrinsics[0, 2]
                v = (intrinsics[1, 1] * corner_camera[1] / corner_camera[2]) + intrinsics[1, 2]
                projected_corners.append((u, v))

        if len(projected_corners) == 8:
            u_vals, v_vals = zip(*projected_corners)
            u_min, u_max = min(u_vals), max(u_vals)
            v_min, v_max = min(v_vals), max(v_vals)

            # Calculate visible portion
            visible_width = max(0, min(u_max, image_width) - max(u_min, 0))
            visible_height = max(0, min(v_max, image_height) - max(v_min, 0))

            total_area = (u_max - u_min) * (v_max - v_min)
            visible_area = visible_width * visible_height

            if total_area > 0:
                visibility_ratio = visible_area / total_area

                if visibility_ratio >= 0.9:
                    visible_furniture.append(label)

    return visible_furniture


def main():
    config = Config()
    pcd_name = config["pre_scanned_graphs"]["high_res"]

    # icp_tform_ground
    aligned_pc_path = config.get_subpath("aligned_point_clouds")
    icp_tform_ground_path = os.path.join(aligned_pc_path, pcd_name, "pose", "icp_tform_ground.txt")
    icp_tform_ground = np.loadtxt(icp_tform_ground_path)

    # furniture_json file
    scene_graph_path = config.get_subpath("scene_graph")
    with open(os.path.join(scene_graph_path, pcd_name, "scene_graph.json"), 'r') as file:
        furniture_json = json.load(file)

    # camera_json files
    ipad_scan_path = config.get_subpath("ipad_scans")
    json_path = os.path.join(ipad_scan_path, pcd_name)
    json_pattern = r"^frame_(\d{5})\.json$"
    json_files = OrderedDict()
    if os.path.exists(json_path) and os.path.isdir(json_path):
        files = os.listdir(json_path)
        files.sort()
        for filename in files:
            file_path = os.path.join(json_path, filename)
            json_match = re.match(json_pattern, str(filename))
            if json_match:
                frame_name = int(json_match.group(1))
                if frame_name in json_files:
                    json_files[frame_name]["json"] = file_path
                else:
                    json_files[frame_name] = {"json": file_path}

    # check camera_json files for furniture objects
    for frame_name, file in json_files.items():
        with open(file["json"], 'r') as file:
            camera_json = json.load(file)
        
        visible_items = check_furniture_visibility(camera_json, furniture_json, icp_tform_ground, image_width=1920, image_height=1440)
        if len(visible_items) != 0:
            print(f"Frame {frame_name} - Visible furniture: {visible_items}")
            
            
def open_o3ds():
    pcd1 = o3d.io.read_point_cloud("obj_cloud_vp.ply")
    pcd2 = o3d.io.read_point_cloud("env_cloud_vp.ply")

    points = np.asarray(pcd1.points)
    filtered_points = points[points[:, 2] > np.min(points[:, 2]) + 0.02] 
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)
    pcd_filtered.colors = o3d.utility.Vector3dVector(np.asarray(pcd1.colors)[points[:, 2] > np.min(points[:, 2]) + 0.02])
    pcd1 = pcd_filtered.voxel_down_sample(0.005)
    pcd2 = pcd2.voxel_down_sample(0.02)
    o3d.visualization.draw_geometries([pcd1], window_name="Point Clouds", width=1920, height=1440)
    o3d.visualization.draw_geometries([pcd2], window_name="Point Clouds", width=1920, height=1440)
    
    gpd_predict_full_grasp(pcd1, pcd2, vis_block=True )
    
def aruco_detect():
    rclpy.init(args=None)
    joint_pose_node = JointPoseController()

    check_object_distance(joint_pose_node)
    
    joint_pose_node.destroy_node()
    rclpy.shutdown()
    
def gripper_position():
    from scipy.spatial.transform import Rotation as R
    rclpy.init(args=None)
    node = rclpy.create_node('transform_manager_node')
    transform_manager = TransformManager(node)
    transform_node = FrameTransformer(transform_manager.tf_buffer)

    tf = transform_node.get_tf_matrix("base_link", "link_grasp_center")
    print("Transformation from base_link to gripper_link:", tf)
    print("Gripper position in base_link frame:", tf[:3, 3])
    print("Gripper orientation in base_link frame:", tf[:3, :3])
    
    rotation_matrix = tf[0:3, 0:3]
    euler = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
    print("Gripper orientation in Euler angles (degrees):", euler) # roll, pitch, yaw
    
    transform_node.destroy_node()
    rclpy.shutdown()
    
def get_couch_pointcloud():
    furnitures = "couch"
    furniture_pcd, env_pcd = get_mask_points(furnitures, Config(), idx=0, vis_block=True)
    front_normal = get_shelf_front_normal(furniture_pcd, "couch")
    
def home_robot_pos():
    rclpy.init(args=None)
    stow_node = StowArmController()
    base_node = BaseController()
    joint_pose_node = JointPoseController()
    
    drive_home(base_node, joint_pose_node, stow_node)
    
    stow_node.destroy_node()
    base_node.destroy_node()
    joint_pose_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    #open_o3ds()
    #aruco_detect()
    #gripper_position()
    #get_couch_pointcloud()
    home_robot_pos()