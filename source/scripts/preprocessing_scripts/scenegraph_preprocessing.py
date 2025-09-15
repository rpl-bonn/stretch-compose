from __future__ import annotations

import json
import numpy as np
import open3d as o3d
import os
import pandas as pd
import time

from utils.coordinates import Pose3D
from utils.preprocessing_utils.drawer_integration import parse_txt
from utils.preprocessing_utils.preprocessing import preprocess_scan
from utils.preprocessing_utils.scene_graph import SceneGraph
from utils.recursive_config import Config
from utils.time import convert_time

# Adaptable
DRAWERS = True

# Configs and Paths
config = Config()
scan_path = config.get_subpath("ipad_scans")
graph_path = config.get_subpath("scene_graph")
aligned_pc_path = config.get_subpath("aligned_point_clouds")
ending = config["pre_scanned_graphs"]["high_res"]
DATA_DIR = config.get_subpath("data")
SCAN_DIR = os.path.join(scan_path, ending)
GRAPH_DIR = os.path.join(graph_path, ending)
TFORM_DIR = os.path.join(aligned_pc_path, ending, "pose")


def compute_pose(points: np.ndarray, centroid: np.ndarray) -> None:
    """
    Compute the pose of an object given its 3D points and centroid.
    This function uses PCA to align the object's points with the global axes and returns the pose matrix.

    Args:
        points (np.ndarray): Array of 3D points representing the object's geometry.
        centroid (np.ndarray): The centroid of the object as a 3D point.
    """
    points_centered = points - centroid
    covariance_matrix = np.cov(points_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    R = eigenvectors
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    object_pose = np.eye(4)
    object_pose[:3, :3] = R
    object_pose[:3, 3] = centroid
    return object_pose.tolist()


def visualize_labeled_mesh(mesh_path: str) -> None:
    """
    Visualize a labeled 3D mesh or point cloud using Open3D.
    This function loads the mesh or point cloud from the specified path and displays it in a 3D viewer.

    Args:
        mesh_path (str): Path to the mesh or point cloud file (e.g., "mesh_labeled.ply").
    """
    # Load the mesh or point cloud
    mesh_labeled = o3d.io.read_point_cloud(mesh_path)
    # Check if it contains color information
    if mesh_labeled.has_colors():
        print("Mesh has vertex colors.")
    else:
        print("Mesh does not have colors. Visualization might be limited.")
    # Visualize using Open3D's viewer
    o3d.visualization.draw_geometries(
        [mesh_labeled],
        window_name="Labeled Mesh Visualization",
        width=800,
        height=600,
        point_show_normal=False
    )
    
    
def add_object_to_scene_graph(obj: str, obj_id: int, center: np.ndarray, conf: float, pcd, location: int, drawer: int) -> None:
    """
    Add an object to the scene graph JSON and save it as an object JSON file.
    This function creates a JSON file for the object and updates the graph.json file with the new object information.

    Args:
        obj (str): Object name
        obj_id (int): Object ID
        center (np.ndarray): 3D coordinates of the object center
        conf (float): Confidence score of the object detection
        pcd (_type_): Point cloud data of the object
        location (int): Location furniture ID of the object
    """
    centroid = center.tolist()
    if pcd != None:
        dim = (np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())).tolist()
        pose = compute_pose(pcd.points, center)
    else:
        dim = [0.0, 0.0, 0.0]
        pose = None
    
    #points = np.asarray(pcd.points).tolist()
    
    # Create objects.json
    node_data = {
        "id": obj_id,
        "label": obj,
        "centroid": centroid,
        "dimensions": dim,
        "pose": pose,
        "drawer": drawer,
        "confidence": conf,
    }
    file_path = os.path.join(GRAPH_DIR, "objects", f"{obj_id}.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(node_data, f, indent=4)
    
    # Extend graph.json
    with open(os.path.join(GRAPH_DIR, "graph.json"), 'r') as file:
        data = json.load(file)
    data["node_ids"].append(obj_id)
    data["node_labels"].append(obj)
    data["connections"][str(obj_id)] = int(location)
    with open(os.path.join(GRAPH_DIR, "graph.json"), 'w') as file:
        json.dump(data, file, indent=4)

        
def update_object_in_scene_graph(obj: str, obj_id: int, center: np.ndarray, conf: float, pcd, location: int, drawer: int) -> None:
    """
    Update an object in the scene graph JSON and save it as an object JSON file.
    This function updates the existing JSON file for the object and updates the graph.json file with the new object information.

    Args:
        obj (str): Object name
        obj_id (int): Object ID
        center (np.ndarray): 3D coordinates of the object center
        conf (float): Confidence score of the object detection
        pcd (_type_): Point cloud data of the object
    """
    centroid = center.tolist()
    #points = np.asarray(pcd.points).tolist()
    
    # Update objects.json
    file_path = os.path.join(GRAPH_DIR, "objects", f"{obj_id}.json")
    with open(file_path, 'r') as f:
        node_data = json.load(f)
    node_data["centroid"] = centroid
    if pcd != None:
        node_data["dimensions"] = (np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())).tolist()
        node_data["pose"] = compute_pose(pcd.points, center)
    node_data["drawer"] = drawer
    node_data["confidence"] = conf 
    with open(file_path, 'w') as f:
        json.dump(node_data, f, indent=4)
    
    # Update connections in graph.json
    with open(os.path.join(GRAPH_DIR, "graph.json"), 'r') as file:
        data = json.load(file)   
    data["connections"][str(obj_id)] = int(location)
    with open(os.path.join(GRAPH_DIR, "graph.json"), 'w') as file:
        json.dump(data, file, indent=4)


def create_scene_graph(scan_path: str, tform_path: str, data_path: str) -> SceneGraph:
    """
    Create a scene graph from the scan data and adapt it.
    This function preprocesses the scan data, removes categories, applies transformations, and builds a scene graph object.

    Args:
        scan_path (str): Path to the directory containing the scan data.
        tform_path (str): Path to the directory containing the icp_tform_ground.txt file.
        data_path (str): Path to the data directory containing the mask3d_label_mapping.csv file.

    Returns:
        SceneGraph: SceneGraph object containing the scene graph data.
    """
    remove = ["door", "window", "doorframe", "radiator", "soap dispenser", "board", "object", "book", "fan", "picture", "backpack"]
    immovable = ["shelf", "bookshelf", "cabinet", "table", "chair", "couch", "armchair", "coffee table", "trash can", "kitchen cabinet", "shelf near door", "end table", "kitchen", "kitchen counter", "ceramic cooktop", "stove", "ceramic hob"]
    
    # Get labels & preprocess scan
    label_map = pd.read_csv(data_path + '/mask3d_label_mapping.csv', usecols=['id', 'category'])
    mask3d_label_mapping = pd.Series(label_map['category'].values, index=label_map['id']).to_dict()
    preprocess_scan(scan_path, drawer_detection=DRAWERS)
    T_ipad = np.load(scan_path + "/aruco_pose.npy")
    
    # Create scene graph
    scene_graph = SceneGraph(label_mapping=mask3d_label_mapping, min_confidence=0.05, immovable=immovable, pose=T_ipad)
    scene_graph.build(scan_path, drawers=DRAWERS)
    scene_graph.remove_categories(remove)
    scene_graph.replace_category("cabinet", "kitchen counter")

    # Transform to Stretch coordinate system
    icp_tform = parse_txt(os.path.join(tform_path, "icp_tform_ground.txt")) # 4x4 transformation matrix of the aruco marker in Stretch coordinate system
    scene_graph.change_coordinate_system(icp_tform)

    return scene_graph


def save_to_json(scene_graph: SceneGraph):
    """
    Save the scene graph data to JSON files.
    This function creates different JSON files for the full graph, furniture, objects, and drawers.

    Args:
        scene_graph (SceneGraph): SceneGraph object containing the scene graph data.
    """
    # Create different files to store scene graph data
    scene_graph.save_full_graph_to_json(os.path.join(GRAPH_DIR, "graph.json"))
    scene_graph.save_furniture_to_json(os.path.join(GRAPH_DIR, "scene.json"))
    scene_graph.save_objects_to_json(os.path.join(GRAPH_DIR, "objects"))
    scene_graph.save_drawers_to_json(os.path.join(GRAPH_DIR, "drawers"))


def main():
    try:
        scenegraph_start = time.time_ns()
        scene_graph = create_scene_graph(SCAN_DIR, TFORM_DIR, DATA_DIR)
        save_to_json(scene_graph)
        scenegraph_end = time.time_ns()
        minutes, seconds = convert_time(scenegraph_end - scenegraph_start)
        print(f"\nSuccessfully created scene_graph (time: {minutes}min {seconds}s).\n")
        #scene_graph.save_visualization(os.path.join(GRAPH_DIR, "visualization.png"), centroids=True, connections=True, labels=True, frame_center=True)
        scene_graph.visualize(labels=True, connections=True, centroids=True, frame_center=True)
    except Exception as e:
        print(f"Error: Failed to create scene graph. {e}")
    
    
if __name__ == "__main__":
    # docker run --gpus all -it -v /home:/home -w /home/stretch/workspace/Test/source/Mask3D rupalsaxena/mask3d_docker:latest -c "python3 mask3d.py --seed 42 --workspace /home/stretch/workspace/stretch-compose/data/ipad_scans/2025_09_12 --pcd && chmod -R 777 /home/stretch/workspace/stretch-compose/data/ipad_scans/2025_09_12"
    # docker run -p 5004:5004 --gpus all -it craiden/yolodrawer:v1.0 python3 app.py
    main()