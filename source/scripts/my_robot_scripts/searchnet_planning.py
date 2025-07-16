import json
import math
import numpy as np
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as R

from stretch_package.stretch_state.frame_transformer import FrameTransformer
from utils.coordinates import Pose3D
from utils.openmask_interface import get_mask_points
from utils.point_clouds import body_planning_front
from utils.recursive_config import Config
from utils.zero_shot_object_detection import get_position_from_head_detection


# Fixed
H_FOV = 42 # horizontal fov 82 | 58 | 42
V_FOV = 69 # vertical fov 140 | 87 | 69

# Adaptable
VIS_BLOCK = False

# Config and Paths
config = Config()


def get_distance_to_shelf(obj: str, index: int|None=None) -> tuple[float, np.ndarray, str, str]:
    """
    Calculate the distance to the shelf for the robot to capture the whole shelf.
    This function retrieves the furniture ID, name, centroid, and dimensions from the object JSON file
    and calculates the distance based on the horizontal and vertical field of view (FOV) of the camera.

    Args:
        obj (str): Object to search for.
        index (int, optional): Location proposal index. Defaults to None.

    Returns:
        tuple: Calculated distance to the shelf, centroid of the furniture, furniture name, and id.
    """
    if index is None:
        # Get furniture id from graph json
        object_path = os.path.join(config.get_subpath("scene_graph"), config["pre_scanned_graphs"]["high_res"], "graph.json")
        with open(object_path, "r") as file:
            object_data = json.load(file)
        
        label_index = object_data["node_labels"].index(obj)
        node_id = object_data["node_ids"][label_index]
        furniture_id = str(object_data["connections"][str(node_id)])
        print(f"Furniture ID: {furniture_id}")
    else:
        # Get furniture id from object json
        object_path = os.path.join(config.get_subpath("scene_graph"), config["pre_scanned_graphs"]["high_res"], "locations", f"{obj.replace(' ', '_')}.json")
        with open(object_path, "r") as file:
            object_data = json.load(file)
        furniture_id = object_data["locations"][index]["furniture_id"]

    # Get furniture name, centroid and dimensions from scene json
    scene_path = os.path.join(config.get_subpath("scene_graph"), config["pre_scanned_graphs"]["high_res"], "scene.json")
    with open(scene_path, "r") as file:
        scene_data = json.load(file)
    furniture_name = scene_data["furniture"][furniture_id]["label"]
    furniture_centroid = scene_data["furniture"][furniture_id]["centroid"]
    furniture_dimensions = scene_data["furniture"][furniture_id]["dimensions"]

    # Calculate robot distance to furniture
    circle_radius_width = (furniture_dimensions[0] + 0.05) / (2 * math.tan(np.radians(H_FOV / 2))) + furniture_dimensions[1] / 2
    circle_radius_height = (furniture_dimensions[2] + 0.05) / (2 * math.tan(np.radians(V_FOV / 2))) + furniture_dimensions[1] / 2
    if furniture_name in ["armchair", "couch", "sofa"]:
        circle_radius_width = (furniture_dimensions[0] -0.15) / (2 * math.tan(np.radians(H_FOV / 2)))
        circle_radius_height = (furniture_dimensions[2] - 0.15) / (2 * math.tan(np.radians(V_FOV / 2)))
    
    return max(circle_radius_width, circle_radius_height), furniture_centroid, furniture_name, furniture_id


def get_shelf_front_normal(furniture_pcd: o3d.geometry.PointCloud, furniture_name: str="") -> np.ndarray:
    """
    Get the normal of the front face of the furniture.
    This function calculates the normal of the front face of the furniture by finding the largest vertical face
    in the oriented bounding box (OBB) of the point cloud.

    Args:
        furniture_pcd (o3d.geometry.PointCloud): Point cloud of the furniture.
        furniture_name (str, optional): Type of furniture, if known. Defaults to "".

    Returns:
        np.ndarray: Normal of the front face of the furniture.
    """
    # Get furniture oriented bounding box
    obb = furniture_pcd.get_minimal_oriented_bounding_box()
    R = obb.R
    extents = obb.extent
    center = obb.center
    
    z_alignment = np.abs(R.T @ np.array([0, 0, 1]))
    vertical_axis = np.argsort(z_alignment)[-2]  # Sort in descending order
    R_new = np.zeros((3, 3))
    
    # Set the vertical axis
    vertical_direction = np.array([0, 0, np.sign(R[2, vertical_axis])])
    R_new[:, vertical_axis] = vertical_direction
    
    # Find most horizontal axis for front direction
    other_axes = [i for i in range(3) if i != vertical_axis]
    horizontal_components = [np.linalg.norm(R[:2, i]) for i in other_axes]
    front_axis = other_axes[np.argmax(horizontal_components)]
    
    # Set front axis
    horizontal_dir = R[:2, front_axis] / np.linalg.norm(R[:2, front_axis])
    R_new[:2, front_axis] = horizontal_dir
    R_new[2, front_axis] = 0
    
    # Set third axis using cross product
    third_axis = [i for i in range(3) if i != vertical_axis and i != front_axis][0]
    R_new[:, third_axis] = np.cross(R_new[:, front_axis], R_new[:, vertical_axis])
        
    # Project points onto the new coordinate axes
    points = np.asarray(furniture_pcd.points)
    points_centered = points - center
    points_rotated = points_centered @ R_new
    min_vals = np.min(points_rotated, axis=0)
    max_vals = np.max(points_rotated, axis=0)
    
    # Calculate new extents and center
    extents_new = max_vals - min_vals
    center_new = center + R_new @ ((min_vals + max_vals) / 2)
    
    if VIS_BLOCK:
        upright_obb = o3d.geometry.OrientedBoundingBox(center_new, R_new, extents_new)
        upright_obb.color = (0, 0, 1)
        o3d.visualization.draw_geometries([furniture_pcd, upright_obb,])

    # Get vertical faces
    vertical_faces = []
    for axis in range(3):
        for direction in [1, -1]:
            # Calculate face normal
            if furniture_name in ["armchair", "couch", "sofa"]:
                normal = R_new[:, axis] * direction
                if np.linalg.norm(center_new) < np.linalg.norm(center_new + normal):
                    normal = -normal
            else:
                normal = R[:, axis] * direction
                if np.linalg.norm(center) < np.linalg.norm(center + normal):
                    normal = -normal
            # Check if normal is roughly horizontal (= vertical face)
            if abs(normal[2]) < 0.1:
                # Calculate face dimensions
                dim1, dim2 = (axis + 1) % 3, (axis + 2) % 3
                if furniture_name in ["armchair", "couch", "sofa"]:
                    area = extents_new[dim1] * extents_new[dim2]
                else:
                    area = extents[dim1] * extents[dim2]
                vertical_faces.append({
                    'normal': normal,
                    'area': area,
                })
    if not vertical_faces:
        raise ValueError("No vertical faces found in shelf structure")
    
    if furniture_name in ["armchair", "couch", "sofa"]:
        front = min(vertical_faces, key=lambda x: x['area'])
    # Select largest vertical face as front
    else:
        front = max(vertical_faces, key=lambda x: x['area'])
    
    if furniture_name in ["coffee table"]:
        front['normal'] = np.array([0.0, 1.0, 0.0])
    
    return front['normal']


def plan_drawer_search(furniture_name: str, center: np.ndarray, drawer_center: Pose3D, dist: float) -> tuple[Pose3D, np.ndarray]:
    """
    Plan the search for the object inside drawers.
    """
    # Get all cabinets/shelfs in the environment
    for idx in range(0, 10):
        furnitures = "kitchen cabinet, shelf"
        furniture_pcd, env_pcd = get_mask_points(furnitures, Config(), idx=idx, vis_block=VIS_BLOCK)
        furniture_center = np.mean(np.asarray(furniture_pcd.points), axis=0)
        # Find correct furniture
        if (np.allclose(furniture_center, center, atol=0.1)):
            print("Shelf/Cabinet found!")
            # Get normal of furniture front face
            front_normal = get_shelf_front_normal(furniture_pcd, furniture_name)
            drawer_center.set_rot_from_direction(-front_normal)
            # Calculate body position in front of furniture
            body_pose = body_planning_front(
                env_pcd,
                target=drawer_center.as_ndarray(),
                furniture_normal=front_normal,
                floor_height_thresh=-0.1,
                min_target_distance=dist,
                max_target_distance=dist+0.2,
                min_obstacle_distance=0.2,
                n=5,
                vis_block=VIS_BLOCK,
            )
            break
    return body_pose, front_normal


def plan_furniture_search(obj: str, index: int|None=None) -> tuple[Pose3D, str, np.ndarray, Pose3D, str]:
    """
    Plan the search for the object in front of the most likely furniture determined by Deepseek.
    This function retrieves the centroid and dimensions of the furniture from the object JSON file,
    calculates the distance to the shelf, finds the front normal of the furniture, 
    and calculates the body pose in front of the furniture.

    Args:
        obj (str): Object to search for.
        index (int, optional): Index of the furniture from the Deepseek result. Defaults to 0.

    Returns:
        tuple[Pose3D, str, np.ndarray, Pose3D]: The centroid of the furniture, the name of the furniture, 
        the front normal of the furniture, the body pose in front of the furniture, and the furniture id.
    """
    # Get necessary distance to shelf
    radius, center, furniture_name, furniture_id = get_distance_to_shelf(obj, index)
    # Get all cabinets/shelfs in the environment
    for idx in range(0, 10):
        if furniture_name == "cabinet":
            furnitures = "kitchen cabinet"
        else:
            furnitures = furniture_name
        furniture_pcd, env_pcd = get_mask_points(furnitures, Config(), idx=idx, vis_block=VIS_BLOCK)
        furniture_center = np.mean(np.asarray(furniture_pcd.points), axis=0)
        # Find correct furniture
        if (np.allclose(furniture_center, center, atol=0.1)):
            print(f"{furniture_name} found!")
            # Get normal of furniture front face
            front_normal = get_shelf_front_normal(furniture_pcd, furniture_name)
            # Calculate body position in front of furniture
            body_pose = body_planning_front(
                env_pcd,
                furniture_center,
                furniture_normal=front_normal,
                min_target_distance=radius,
                max_target_distance=radius+0.2,
                min_obstacle_distance=0.4,
                n=5,
                vis_block=VIS_BLOCK,
            )
            break

        
    return Pose3D(furniture_center), furniture_name, front_normal, body_pose, furniture_id


def plan_object_search(
    transform_node: FrameTransformer, detection_dict: dict, front_normal: np.ndarray, pcd: o3d.geometry.PointCloud, furniture_id: str
) -> tuple[Pose3D, Pose3D, o3d.geometry.PointCloud]:
    """
    Calculate the position of the object and a body pose in front of it.
    This function retrieves the object center and dimensions from the detection dictionary, removes points from the point cloud around the object location,
    and calculates a body pose in front of the object.

    Args:
        transform_node (FrameTransformer): ROS node for frame transformation
        detection_dict (dict): Dictionary containing detection information
        front_normal (np.ndarray): Normal vector of the front of the furniture
        pcd (o3d.geometry.PointCloud): Point cloud of the environment

    Returns:
        tuple[Pose3D, Pose3D, o3d.geometry.PointCloud]: A tuple containing the object center pose, robot body pose, and updated point cloud.
    """
    center, width, height = get_position_from_head_detection(detection_dict, transform_node)
    center.set_rot_from_direction(-front_normal)
    center_np = center.as_ndarray()
    
    # Remove points from environment point cloud around the object location
    distances = np.linalg.norm(np.asarray(pcd.points) - center_np, ord=2, axis=1)
    radius = max(width, height)
    pcd = pcd.select_by_index(np.where(distances > radius)[0].tolist())
    
    if VIS_BLOCK:
        x = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        x.translate(center_np.reshape(3, 1))
        x.paint_uniform_color((1, 0, 0))
        o3d.visualization.draw_geometries([x, pcd]) 
    
    # Shift object coordinates to furniture edge in normal direction   
    scene_path = os.path.join(config.get_subpath("scene_graph"), config["pre_scanned_graphs"]["high_res"], "scene.json")
    with open(scene_path, "r") as file:
        scene_data = json.load(file)   
    furniture_center = scene_data["furniture"][furniture_id]["centroid"]
    furniture_dimensions = scene_data["furniture"][furniture_id]["dimensions"]
    
    object_dist_along_normal = np.dot(center_np - furniture_center, front_normal)
    edge_dist_along_normal = furniture_dimensions[1] / 2
    if edge_dist_along_normal < 0.3:
        edge_dist_along_normal += 0.08
    if furniture_id == "8":
        edge_dist_along_normal += 0.15
    
    target_pos = center_np + (edge_dist_along_normal - object_dist_along_normal) * front_normal
    print(f"Target position: {target_pos}")
    print(edge_dist_along_normal, object_dist_along_normal)
    
    # Move robot closer to object
    body_pose = body_planning_front(
        pcd,
        target=target_pos,
        furniture_normal=front_normal,
        floor_height_thresh=-0.1,
        body_height=1.0,
        min_target_distance=0.4,
        max_target_distance=0.4,
        min_obstacle_distance=0.3,
        n=5,
        vis_block=VIS_BLOCK
    )
    print(f"Body pose: {body_pose.as_ndarray()}")
    
    return center, body_pose, pcd

def filter_drawers(in_scene_graph: bool, graph_data: dict, connections: dict, checked_furniture_ids: list, OBJECT: str) -> None:
    graph_path = config.get_subpath("scene_graph")      
    ending = config["pre_scanned_graphs"]["high_res"]
    GRAPH_DIR = os.path.join(graph_path, ending)
    
    # Only check drawers inside the furniture already checked
    possible_drawers = [id for id, label in zip(graph_data["node_ids"], graph_data["node_labels"]) if "drawer" in label and str(connections[str(id)]) in checked_furniture_ids]
    sorted_drawers = sorted(possible_drawers, key=lambda x: checked_furniture_ids.index(str(connections[str(x)])))
    print(f"Possible drawers for {OBJECT}: {sorted_drawers}")
    
    # Only check drawers that are bigger than the known object size
    if in_scene_graph:
        obj_id = graph_data["node_ids"][graph_data["node_labels"].index(OBJECT)]
        with open(os.path.join(GRAPH_DIR, "objects", f"{obj_id}.json"), "r") as file:
            object_data = json.load(file)
        fitting_drawers = []
        for drawer_id in sorted_drawers:
            with open(os.path.join(GRAPH_DIR, "drawers", f"{drawer_id}.json"), "r") as file:
                drawer_data = json.load(file)
            drawer_dimensions = drawer_data["dimensions"]
            drawer_diagonal = np.linalg.norm(drawer_dimensions[0:2])
            if drawer_dimensions[2]>object_data["dimensions"][2] and drawer_diagonal>max(object_data["dimensions"][0:2]):
                fitting_drawers.append(drawer_id)           
    else:
        fitting_drawers = sorted_drawers.copy()
    print(f"Fitting drawers for {OBJECT}: {fitting_drawers}")
    
    return fitting_drawers


if __name__ == "__main__":
    plan_furniture_search("book", 0)
