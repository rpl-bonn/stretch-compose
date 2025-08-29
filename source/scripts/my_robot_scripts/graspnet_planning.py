import copy
import numpy as np
import time


from gpd.gpd_client_api import predict_full_grasp as gpd_predict_full_grasp
from utils import graspnet_interface, openmask_interface
from utils.coordinates import Pose3D
from utils.point_clouds import body_planning, get_radius_env_cloud, visualize_environment
from utils.recursive_config import Config
from utils.robot_utils.advanced_movement import adapt_body, adapt_grasp, optimize_joints
from utils.time import convert_time
from utils.user_input import get_wanted_item_mask3d


# Fixed
RADIUS = 1.0
RESOLUTION = 36
LAM_BODY = 0.05
LAM_ALIGNMENT = 0.02
LAM_OBJECT = 0.5

# Adaptable
VIS_BLOCK = True
OBJECT = 'bottle'

# Config and Paths
config = Config()


def plan_poses(obj: str, vis_block: bool=False) -> tuple[Pose3D, Pose3D]:
    """
    Plan the poses for the robot body and the grasp of the object. 
    The function localizes the object in the environment point cloud, predicts the grasp using GraspNet,
    and optimizes the robot's body pose and joint angles for the grasp.

    Args:
        obj (str): object to detect and grasp.
        vis_block (bool, optional): True if visualizations should be shown, False otherwise. Defaults to False.

    Returns:
        tuple[Pose3D, Pose3D]: robot body pose and grasp pose.
    """
    # Object localization
    try:
        mask_points_start = time.time_ns()
        obj_cloud, environment_cloud = openmask_interface.get_mask_points(
            obj, 
            config, 
            vis_block=vis_block
        )
        lim_env_cloud = get_radius_env_cloud(
            obj_cloud, 
            environment_cloud, 
            RADIUS
        )
        obj_points = np.asarray(obj_cloud.points)
        
        size = np.max(obj_points, axis=0)-np.min(obj_points, axis=0)
        print(f"Object size: {size}")
        center = obj_points.mean(axis=0)
        print(f"Object center: {center}")
        
        mask_points_end = time.time_ns()
        minutes, seconds = convert_time(mask_points_end - mask_points_start)
        print(f"Successfully detected object in point cloud (time: {minutes}min {seconds}s).\n")
    except Exception:
        print("Error: Failed detecting object in point cloud.")

    # Grasp prediction
    grasp_predicted = False
    try:
        predict_grasp_start = time.time_ns()
        tf_matrices, widths, grasp_scores = gpd_predict_full_grasp( #graspnet_interface.predict_full_grasp(
            obj_cloud,
            lim_env_cloud,
            config,
            rotation_resolution=RESOLUTION,
            top_n=3,
            n_best=30,
            vis_block=True,
        )
        obj_center = Pose3D.from_matrix(tf_matrices[0])
        predict_grasp_end = time.time_ns()
        minutes, seconds = convert_time(predict_grasp_end - predict_grasp_start)
        grasp_predicted = True
        print(f"Successfully predicted grasp (time: {minutes}min {seconds}s).\n")
    except Exception as e:
        print(f"Error: Failed predicting grasp. {e}")

    # Body Planning
    if not grasp_predicted:
        obj_center = Pose3D(obj_points.mean(axis=0))
    try:
        body_planning_start = time.time_ns()
        body_scores = body_planning(
            environment_cloud,
            obj_center,
            resolution=RESOLUTION,
            nr_circles=2,
            min_distance=0.8,
            max_distance=0.9,
            lambda_distance=0.4,
            floor_height_thresh=0.1,
            n_best=15,
            body_height=1.0,
            vis_block=True,
        )
        body_planning_end = time.time_ns()
        minutes, seconds = convert_time(body_planning_end - body_planning_start)
        print(f"{body_scores=}")
        print(f"Successfully planned body (time: {minutes}min {seconds}s).\n")
    except Exception:
        print("Error: Failed planning body.")

    # Joint Optimization
    if grasp_predicted:
        try:
            joint_optimization_start = time.time_ns()
            best_grasp, best_pose, _ = optimize_joints(
                obj_center,
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
        grasp_pose_new = Pose3D(adapt_grasp(best_pose, best_grasp.as_ndarray()))
        print(f"{grasp_pose_new=}") 
    
    # Visualization
    if True:
        visualize_environment(obj_cloud, environment_cloud, best_pose)
    
    return body_pose_distanced, grasp_pose_new, max(size[0], size[1])
    
    
if __name__ == "__main__":
    # docker run --net=bridge --mac-address=02:42:ac:11:00:02 -p 5000:5000 --gpus all -it craiden/graspnet:v1.0 python3 app.py
    if OBJECT is None:
        obj = get_wanted_item_mask3d()
    else:
        obj = OBJECT
    body_pose_distanced, grasp_pose_new, obj_width = plan_poses(obj, vis_block=VIS_BLOCK)
