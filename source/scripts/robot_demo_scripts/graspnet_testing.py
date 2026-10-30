import copy
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from typing import List

from utils import graspnet_interface, openmask_interface
from utils.coordinates import Pose3D
from utils.point_clouds import get_radius_env_cloud
from utils.recursive_config import Config
from gpd.gpd_client_api import predict_full_grasp as gpd_predict_full_grasp

# Fixed
RADIUS = 1.0

#Adaptable
VIS_BLOCK = False
OBJECT = 'shark plushy'
OBJECTS = ["green watering can", "watering can", "box", "heater", "bin", "plant", "shark plushy", "shelf", "armchair", "couch", 
           "stove", "coffee table", "bottle", "kitchen cabinet", "cat plushy", "cup", "milk", "herbs", "image frame", "kallax", 
           "bowl", "cooking utensils", "chair", "book", "tennis ball", "plate", "pan", "pink folder","football", "cooking pot", ]

# Config and Paths
config = Config()


def find_all_items() -> None:
    for obj in OBJECTS:
        openmask_interface.get_mask_points(obj, Config(), vis_block=True)


def find_shelves() -> tuple[List[np.ndarray], List[Pose3D]]: 
    for idx in range(0,10):
        cabinet_pcd, _ = openmask_interface.get_mask_points("cabinet", Config(), idx=idx, vis_block=True)
        cabinet_pcd, _ = openmask_interface.get_mask_points("shelf", Config(), idx=idx, vis_block=True) 
        cabinet_pcd, _ = openmask_interface.get_mask_points("bookshelf", Config(), idx=idx, vis_block=True)
        cabinet_pcd, _ = openmask_interface.get_mask_points("couch", Config(), idx=idx, vis_block=True)  
        cabinet_pcd, _ = openmask_interface.get_mask_points("coffee table", Config(), idx=idx, vis_block=True) 


def create_coordinate_frame(transform_matrix, size=0.05) -> o3d.geometry.TriangleMesh:
    """Create a coordinate frame at the given pose with the given size."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(transform_matrix)
    return frame

def create_gripper_model(transform_matrix, width, color=[0, 1, 0]) -> o3d.geometry.TriangleMesh:
    """Create a simple gripper model at the given pose with the given width and color."""
    # Create base of the gripper
    base_height = 0.02
    base_width = 0.04
    base = o3d.geometry.TriangleMesh.create_box(width=base_height, height=base_width, depth=base_width)
    
    # Move base to origin
    base.translate([-base_height/2, -base_width/2, -base_width/2])
    
    # Create fingers
    finger_width = 0.01
    finger_height = 0.04
    finger_depth = 0.02
    
    left_finger = o3d.geometry.TriangleMesh.create_box(width=finger_height, height=finger_width, depth=finger_depth)
    right_finger = o3d.geometry.TriangleMesh.create_box(width=finger_height, height=finger_width, depth=finger_depth)
    
    # Position the fingers
    left_finger.translate([0, -width/2 - finger_width/2, -finger_depth/2])
    right_finger.translate([0, width/2 - finger_width/2, -finger_depth/2])
    
    # Combine into one mesh
    gripper = base
    gripper += left_finger
    gripper += right_finger
    
    # Paint the gripper
    gripper.paint_uniform_color(color)
    
    # Transform the gripper to the grasp pose
    gripper.transform(transform_matrix)
    
    return gripper


def visualize_grasps(obj_cloud: o3d.geometry.PointCloud, env_cloud: o3d.geometry.PointCloud, tf_matrices: np.ndarray, widths: np.ndarray, scores: np.ndarray, save_to: str) -> o3d.geometry.PointCloud:
    """
    Visualize the detected grasp poses as coordinate frames in Open3D.
    
    Args:
        obj_cloud: Point cloud of the object to grasp
        env_cloud: Point cloud of the environment
        tf_matrices: Transformation matrices of the grasp poses
        widths: Widths of the gripper for each grasp
        scores: Scores for each grasp
        save_path: Path to save the visualization point cloud (optional)
        
    Returns:
        o3d.geometry.PointCloud: Combined point cloud with grasp visualizations
    """
    # Create a new point cloud to visualize
    visualized_cloud = o3d.geometry.PointCloud()
    
    # Add the object and environment clouds
    visualized_cloud += obj_cloud
    visualized_cloud += env_cloud
    
    # Create a list to store all grasp frames for visualization
    grasp_frames = []
    
    # Color for the different grasps (from red=best to blue=worst)
    color_map = plt.cm.jet
    
    # Normalize scores for coloring
    if len(scores) > 0:
        score_min = min(scores)
        score_max = max(scores)
        score_range = score_max - score_min if score_max > score_min else 1.0
    
    # Create a coordinate frame for each grasp
    for i, (transform, width, score) in enumerate(zip(tf_matrices, widths, scores)):         
        # Create a coordinate frame
        ## Adjust size as needed and use the translation part of the transform
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=transform[:3, 3])
        
        # Apply rotation from transform
        frame.rotate(transform[:3, :3], center=transform[:3, 3])
        
        # Add the frame to the list
        grasp_frames.append(frame)
        
        # Create grasp width visualization (a line between the fingers)
        finger_width = width
        
        # Calculate finger positions in local coordinate frame
        ## Assuming the gripper closes along the x-axis
        left_finger = transform @ np.array([-finger_width/2, 0, 0, 1])
        right_finger = transform @ np.array([finger_width/2, 0, 0, 1])
        
        # Create points for fingers
        finger_points = np.vstack([left_finger[:3], right_finger[:3]])
        finger_cloud = o3d.geometry.PointCloud()
        finger_cloud.points = o3d.utility.Vector3dVector(finger_points)
        
        # Color based on score (normalized) (red = best, blue = worst)
        if len(scores) > 1:
            norm_score = (score - score_min) / score_range
            color = color_map(norm_score)[:3]  # Get RGB from colormap
        else:
            color = [1, 0, 0]  # Red for single grasp
            
        # Set the color for finger points
        finger_cloud.paint_uniform_color(color)
        
        # Add to visualization
        visualized_cloud += finger_cloud
        
        
    vis_geometries = []
    env_cloud_colored = copy.deepcopy(env_cloud)
    if not env_cloud_colored.has_colors():
        env_cloud_colored.paint_uniform_color([0.8, 0.8, 0.8])
    vis_geometries.append(env_cloud_colored)
    
    # Make object cloud blue
    obj_cloud_colored = copy.deepcopy(obj_cloud)
    if not obj_cloud_colored.has_colors():
        obj_cloud_colored.paint_uniform_color([0.0, 0.0, 1.0])
    vis_geometries.append(obj_cloud_colored)
    
    for i, (transform, width, score) in enumerate(zip(tf_matrices, widths, scores)):
            print(f"Adding grasp {i+1} with score {score}")
            
            # Create coordinate frame
            #frame = create_coordinate_frame(transform, size=0.05)
            #vis_geometries.append(frame)
            
            # Create gripper model
            # Map score to a color (red for high score, blue for low score)
            if len(scores) > 1:
                score_min = min(scores)
                score_max = max(scores)
                score_range = score_max - score_min if score_max > score_min else 1.0
                norm_score = (score - score_min) / score_range
                color = [norm_score, 0.5*(1-norm_score), 1-norm_score]  # red to blue
            else:
                color = [1, 0, 0]  # Red for single grasp
            
            gripper = create_gripper_model(transform, width, color)
            vis_geometries.append(gripper)
        
    # Show visualization
    #o3d.visualization.draw_geometries(vis_geometries, window_name="Grasp Visualization", width=1000, height=800)
    
    # save vis geometries to a point cloud
    save_path = f"/home/ws/data/images/{save_to}.ply"
    combined_points = []
    combined_colors = []

    for g in vis_geometries:
        if isinstance(g, o3d.geometry.PointCloud):
            combined_points.append(np.asarray(g.points))
            if g.has_colors():
                combined_colors.append(np.asarray(g.colors))
        elif isinstance(g, o3d.geometry.TriangleMesh):
            sampled_pcd = g.sample_points_uniformly(number_of_points=500)
            combined_points.append(np.asarray(sampled_pcd.points))
            if sampled_pcd.has_colors():
                combined_colors.append(np.asarray(sampled_pcd.colors))

    vis_cloud = o3d.geometry.PointCloud()
    vis_cloud.points = o3d.utility.Vector3dVector(np.vstack(combined_points))
    if combined_colors:
        vis_cloud.colors = o3d.utility.Vector3dVector(np.vstack(combined_colors))
    o3d.io.write_point_cloud(save_path, vis_cloud)
    
    return visualized_cloud


def grasping_test():
    # Object localization
    try:
        obj_cloud, environment_cloud = openmask_interface.get_mask_points(OBJECT, config, vis_block=VIS_BLOCK)
        lim_env_cloud = get_radius_env_cloud(obj_cloud, environment_cloud, RADIUS)
        
        #o3d.io.write_point_cloud("obj_cloud.ply", obj_cloud)
        #o3d.io.write_point_cloud("env_cloud.ply", lim_env_cloud)
        
        print(f"Successfully detected object in point cloud.\n")
    except Exception as e:
        print(f"Error: Failed detecting object in point cloud. {e}")

    # Grasp prediction
    try:
        tf_matrices, widths, scores = gpd_predict_full_grasp( #graspnet_interface.predict_full_grasp(
            obj_cloud,
            lim_env_cloud,
            rotation_resolution=24,
            top_n=3,
            n_best=60,
        )
        print(f"Found {len(scores)} grasp candidates")
        print(tf_matrices[0], scores, widths)
        print(f"Successfully predicted grasp.\n")
    except Exception as e:
        print(f"Error: Failed predicting grasp. {e}")


if __name__ == "__main__":
    find_all_items()
    centers = find_shelves()
    #grasping_test()