import numpy as np
import open3d as o3d
import time
from typing import Tuple
from scripts.point_cloud_scripts.point_cloud_preprocessing import convert_time
from scipy.spatial.transform import Rotation
from utils.recursive_config import Config
from utils.coordinates import Pose3D
from utils import graspnet_interface
from utils.user_input import get_wanted_item_mask3d
from utils.openmask_interface import get_mask_points
from utils.point_clouds import get_radius_env_cloud
from utils.point_clouds import body_planning
from utils.point_clouds import body_planning_mult_furthest
from robot_utils.advanced_movement import adapt_grasp

RADIUS = 0.75
RESOLUTION = 16
LAM_BODY = 0.05
LAM_ALIGNMENT = 0.02
LAM_ITEM = 0.5

def optimize_joints(target: Pose3D, tf_matrices: np.ndarray, widths: np.ndarray, grasp_scores: np.ndarray, body_scores: list[tuple[Pose3D, float]], 
                    lambda_body: float = 0.5, lambda_alignment: float = 1.0, temperature: float = 0.2,) -> Tuple[Pose3D, Pose3D, float]: 
    if not body_scores:
        raise ValueError("body_scores is empty; at least one pose is required.")
    
    nr_grasps = tf_matrices.shape[0]
    nr_poses = len(body_scores)
    # matrix is nr_grasps x nr_poses 
    grasp_scores = grasp_scores.reshape((nr_grasps, 1))
    grasp_score_mat = np.tile(grasp_scores, (1, nr_poses))
    pose_scores = np.asarray([score for (_, score) in body_scores]).reshape((1, nr_poses))
    pose_score_mat = np.tile(pose_scores, (nr_grasps, 1))
    
    grasp_directions_np = np.stack([Pose3D.from_matrix(tf).direction() for tf in tf_matrices], axis=0)
    body_coordinates_np = np.stack([pose.coordinates for (pose, _) in body_scores], axis=1)
    body_to_targets_np = target.coordinates.reshape((3, 1)) - body_coordinates_np
    body_coordinates_norm = body_to_targets_np / np.linalg.norm(body_to_targets_np, axis=0, keepdims=True)

    alignment_mat = grasp_directions_np @ body_coordinates_norm
    alignment_mat_tanh = np.tanh(alignment_mat * temperature)
    joint_matrix = (grasp_score_mat + lambda_body * pose_score_mat + lambda_alignment * alignment_mat_tanh)

    grasp_argmax, pose_argmax = np.unravel_index(np.argmax(joint_matrix), joint_matrix.shape)
    print(f"{grasp_argmax=}", f"{pose_argmax=}")
    
    best_grasp = Pose3D.from_matrix(tf_matrices[grasp_argmax])
    best_pose = body_scores[pose_argmax][0]
    best_width = widths[grasp_argmax]  
    return best_grasp, best_pose, best_width


def adapt_body(best_pose: Pose3D, best_grasp: Pose3D):
    # Get extra offset from body-grasp distance
    print(f"body-grasp dist: {np.linalg.norm(best_pose.to_dimension(2).coordinates - best_grasp.to_dimension(2).coordinates)}")
    extra_offset = 0.6-np.linalg.norm(best_pose.to_dimension(2).coordinates - best_grasp.to_dimension(2).coordinates)
    # Get direction
    direction = best_pose.to_dimension(2).coordinates - best_grasp.to_dimension(2).coordinates
    unit_direction = direction / np.linalg.norm(direction)
    # Add offset to direction
    body_pose_distanced = best_pose.copy()
    body_pose_distanced.coordinates = body_pose_distanced.coordinates + np.append(unit_direction, 0.0) * extra_offset
    return body_pose_distanced


def visualize_environment(item_cloud, environment_cloud, body_pose_distanced: Pose3D):
    item_cloud.paint_uniform_color([1, 0, 1])
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    sphere.paint_uniform_color([1, 0, 0])
    sphere.translate(body_pose_distanced.coordinates)
    o3d.visualization.draw_geometries([item_cloud, environment_cloud, sphere])


def find_all_items():
    ITEMS = ["heater", "bin", "plant", "shark plushy", "shelf", "yellow armchair", "kallax",
             "bottle", "mannequin", "kitchen cabinet", "door", "cat plushy"]
    # "owl plushy", "penguin plushy", "rubiks cube", "book", "table", "tennis ball", "image frame"
    for item in ITEMS:
        get_mask_points(item, Config(), vis_block=True)


def find_shelves(vis_block=False):
    cabinet_centers = []
    body_poses = []
    
    for idx in range(1,4):
        cabinet_pcd, env_pcd = get_mask_points("cabinet, shelf", Config(), idx=idx, vis_block=vis_block)
        cabinet_center = np.mean(np.asarray(cabinet_pcd.points), axis=0)
        cabinet_centers.append(cabinet_center)
        body_pose = body_planning_mult_furthest(
            env_pcd,
            cabinet_center,
            min_target_distance=1.75,
            max_target_distance=2.25,
            min_obstacle_distance=0.5,
            n=4,
            vis_block=vis_block,
        )
        body_poses.append(body_pose)
        
    print(f"{cabinet_centers=}")
    print(f"{body_poses=}")
    
    return cabinet_centers, body_poses


def find_item(item, vis_block=False):
    config = Config()

    # Item localization
    try:
        mask_points_start = time.time_ns()
        item_cloud, environment_cloud = get_mask_points(item, config, vis_block=vis_block)
        lim_env_cloud = get_radius_env_cloud(item_cloud, environment_cloud, RADIUS)
        item_points = np.asarray(item_cloud.points)
        item_size = np.max(item_points, axis=0)-np.min(item_points, axis=0)
        print(f"{item_size}")
        mask_points_end = time.time_ns()
        minutes, seconds = convert_time(mask_points_end - mask_points_start)
        print(f"Successfully detected item in point cloud (time: {minutes}min {seconds}s).\n")
    except Exception:
        print("Error: Failed detecting item in point cloud.")

    # Grasp prediction
    try:
        predict_grasp_start = time.time_ns()
        tf_matrices, widths, grasp_scores = graspnet_interface.predict_full_grasp(
            item_cloud,
            lim_env_cloud,
            config,
            rotation_resolution=24,
            top_n=3,
            n_best=60,
            vis_block=vis_block,
        )
        item_center = Pose3D.from_matrix(tf_matrices[0])
        predict_grasp_end = time.time_ns()
        minutes, seconds = convert_time(predict_grasp_end - predict_grasp_start)
        print(f"Successfully predicted grasp (time: {minutes}min {seconds}s).\n")
    except Exception:
        print("Error: Failed predicting grasp.")

    # Body Planning
    try:
        body_planning_start = time.time_ns()
        body_scores = body_planning(
            environment_cloud,
            item_center,
            resolution=24,
            nr_circles=2,
            min_distance=0.5,
            max_distance=0.6,
            lambda_distance=LAM_ITEM,
            floor_height_thresh=0.1,
            n_best=15,
            body_height=1.0,
            vis_block=vis_block,
        )
        body_planning_end = time.time_ns()
        minutes, seconds = convert_time(body_planning_end - body_planning_start)
        print(f"{body_scores=}")
        print(f"Successfully planned body (time: {minutes}min {seconds}s).\n")
    except Exception:
        print("Error: Failed planning body.")

    # Joint Optimization
    try:
        joint_optimization_start = time.time_ns()
        best_grasp, best_pose, width = optimize_joints(
            item_center,
            tf_matrices,
            widths,
            grasp_scores,
            body_scores,
            lambda_body=LAM_BODY,
            lambda_alignment=LAM_ALIGNMENT,
        )
        joint_optimization_end = time.time_ns()
        minutes, seconds = convert_time(joint_optimization_end - joint_optimization_start)
        print(f"{best_grasp=}", f"\n{best_pose=}")
        print(f"Successfully optimized joints (time: {minutes}min {seconds}s).\n")
    except Exception:
        print("Error: Failed optimizing joints.")

    # Adapt body pose
    body_pose_distanced = adapt_body(best_pose, best_grasp)
    print(f"{body_pose_distanced.to_dimension(2)=}")
    
    # Adapt grasp pose
    grasp_pose_new = best_grasp.copy() #adapt_grasp(best_pose, best_grasp)
    print(f"{grasp_pose_new=}") 
    
    # Visualization
    if vis_block:
        visualize_environment(item_cloud, environment_cloud, best_pose)
    
    return body_pose_distanced, grasp_pose_new
    
if __name__ == "__main__":
    #find_all_items()
    #centers, body_poses = find_shelves(True)
    
    # DON'T FORGET TO RUN: docker run -p 5000:5000 --gpus all -it craiden/graspnet:v1.0 python3 app.py
    item = 'bottle' #get_wanted_item_mask3d()
    body_pose_distanced, grasp_pose_new = find_item(item, True)
