
import os
from utils.recursive_config import Config
from scripts.temp_scripts.deepseek_exploration import ask_deepseek
import json
import math
import numpy as np
from utils.point_clouds import body_planning_front
from utils.openmask_interface import get_mask_points
import open3d as o3d


OBJECT = 'watering can' # object to search for
H_FOV = 82 # horizontal fov of robot camera
V_FOV = 140 # vertical fov of robot camera

config = Config()

def get_distance_to_shelf(index: int=0) -> tuple:
    '''
    Get the distance to the front of the furniture for the robot to capture whole shelf.
    
    :param index: Index of the shelf/cabinet from the Deepseek result.
    :return: Tuple of the calculated distance and the centroid of the furniture.
    '''
    # get furniture id from object json
    object_path = os.path.join(config.get_subpath("scene_graph"), config["pre_scanned_graphs"]["high_res"], f"{OBJECT.replace(' ', '_')}.json")
    with open(object_path, "r") as file:
        object_data = json.load(file)
    furniture_id = object_data[OBJECT.replace("_", " ")][index]["furniture_id"]

    # get furniture centroid and width from scene json
    scene_path = os.path.join(config.get_subpath("scene_graph"), config["pre_scanned_graphs"]["high_res"], "scene_graph.json")
    with open(scene_path, "r") as file:
        scene_data = json.load(file)
    furniture_centroid = scene_data["nodes"][furniture_id]["centroid"]
    furniture_dimensions = scene_data["nodes"][furniture_id]["dimensions"]

    # calculate robot distance to furniture
    circle_radius_width = (furniture_dimensions[0] + 0.1) / (2 * math.tan(np.radians(H_FOV / 2))) + furniture_dimensions[1] / 2
    circle_radius_height = (furniture_dimensions[2] + 0.1) / (2 * math.tan(np.radians(V_FOV / 2))) + furniture_dimensions[1] / 2
    print(circle_radius_width, circle_radius_height, furniture_dimensions[0])
    return max(circle_radius_width, circle_radius_height), furniture_centroid


def get_shelf_front(cabinet_pcd: o3d.geometry.PointCloud) -> np.ndarray:
    '''
    Get the normal of the front face of the furniture.
    
    :param cabinet_pcd: PointCloud of the shelf/cabinet.
    :return: Normal of the front face.
    '''
    # get furniture oriented bounding box
    obb = cabinet_pcd.get_oriented_bounding_box()
    R = obb.R
    extents = obb.extent

    # get vertical faces
    vertical_faces = []
    for axis in range(3):
        for direction in [1, -1]:
            # calculate face normal
            normal = R[:, axis] * direction
            # check if normal is roughly horizontal (= vertical face)
            if abs(normal[2]) < 0.1:
                # calculate face dimensions
                dim1 = (axis + 1) % 3
                dim2 = (axis + 2) % 3
                area = extents[dim1] * extents[dim2]
                vertical_faces.append({
                    'normal': normal,
                    'area': area,
                })
    if not vertical_faces:
        raise ValueError("No vertical faces found in shelf structure")
    
    # select largest vertical face as front
    front = max(vertical_faces, key=lambda x: x['area'])
    return front['normal']
    

def get_circle_pose(index: int=0):
    '''
    Get the pose in front of the most likely furniture determined by Deepseek.
    
    :param index: Index of the furniture from the Deepseek result.
    :return: Tuple of the centroid of the furniture and the body pose.
    '''
    # get necessary distance to shelf
    radius, center = get_distance_to_shelf(index)
    radius = max(0.8, radius)
    # get all cabinets/shelfs in the environment
    for idx in range(1, 5):
        furnitures = "kitchen cabinet, shelf"
        cabinet_pcd, env_pcd = get_mask_points(furnitures, Config(), idx=idx, vis_block=False)
        cabinet_center = np.mean(np.asarray(cabinet_pcd.points), axis=0)
        # find correct cabinet/shelf
        if (np.allclose(cabinet_center, center, atol=0.1)):
            print("Shelf found!")
            # get normal of cabinet/shelf front face
            front_normal = get_shelf_front(cabinet_pcd)
            # calculate body position in front of cabinet/shelf
            body_pose = body_planning_front(
                env_pcd,
                cabinet_center,
                shelf_normal=front_normal,
                min_target_distance=radius,
                max_target_distance=radius+0.2,
                min_obstacle_distance=0.4,
                n=5,
                vis_block=True,
            )
    
    return cabinet_center, body_pose


def main():
    # pipeline to get object via pointcloud segmentation
    # ... robot dependant ...
    # pipeline after object has not been found at original location
    ask_deepseek(OBJECT)
    center, pose = get_circle_pose(0)  # to get pose in front of most likely (=index 0) furniture determined by deepseek
    # drive to pose in front of shelf
    # take picture of shelf with robot
    # ask GPT if object is in shelf


if __name__ == "__main__":
    main()
