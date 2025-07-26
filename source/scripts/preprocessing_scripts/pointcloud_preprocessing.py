import os
import time
from utils.preprocessing_utils import full_merge
from utils.preprocessing_utils import full_align
from utils.openmask_interface import get_mask_clip_features
from utils.recursive_config import Config
from utils.time import convert_time
import open3d as o3d

config = Config()

def show_point_cloud(path, name="Point Cloud") -> None:
    """
    Shows a point cloud in Open3D.
    :param path: Path where the point cloud is located.
    :param name: Window name for Open3D.
    """
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    pcd = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd, axis], window_name=name)


def merge_clouds():
    """
    Merges point clouds from autowalk scan to one point cloud.
    """
    try:
        point_clouds_start = time.time_ns()
        full_merge.main()
        point_clouds_end = time.time_ns()
        minutes, seconds = convert_time(point_clouds_end - point_clouds_start)
        print(f"\nSuccessfully created point_clouds (time: {minutes}min {seconds}s).\n")
    except FileNotFoundError:
        print("Error: The autowalk_scans/low_res folder does not exist.")


def align_clouds():
    """
    Aligns the high_res and low_res point clouds to the robot map frame.
    """
    try:
        align_start = time.time_ns()
        full_align.main()
        align_end = time.time_ns()
        minutes, seconds = convert_time(align_end - align_start)
        print(f"\nSuccessfully created aligned_point_clouds (time: {minutes}min {seconds}s).\n")
    except FileNotFoundError:
        print("Error: The ipad_scans/high_res folder does not exist.")


def get_mask():
    """
    Runs the OpenMask3D Segmentation.
    """
    try:
        openmask_features_start = time.time_ns()
        get_mask_clip_features()
        openmask_features_end = time.time_ns()
        minutes, seconds = convert_time(openmask_features_end - openmask_features_start)
        print(f"\nSuccessfully created openmask_features (time: {minutes}min {seconds}s).\n")
    except ConnectionError:
        print("Error: Failed to establish connection to port 5001.")


def show_clouds():
    """
    Shows the ipad, autowalk, and aligned point clouds.
    """
    base_data_path = config.get_subpath("data")
    scan_path = os.path.join(str(base_data_path), "ipad_scans", f'{config["pre_scanned_graphs"]["high_res"]}', 'mesh.ply')
    autowalk_path = os.path.join(str(base_data_path), "merged_point_clouds", f'{config["pre_scanned_graphs"]["low_res"]}.ply')
    aligned_path = os.path.join(str(base_data_path), "aligned_point_clouds", f'{config["pre_scanned_graphs"]["high_res"]}', 'scene.ply')

    show_point_cloud(scan_path, name="IPad scan")
    show_point_cloud(autowalk_path, name="Autowalk scan")
    show_point_cloud(aligned_path, name="Aligned point cloud")
    
def show_object_and_grasps():
    """
    Shows the object point cloud, limited environment point cloud and grasps.
    """
    show_point_cloud("/home/ws/data/images/viewpoints/env_cloud_vp.ply", name="Env cloud")
    show_point_cloud("/home/ws/data/images/viewpoints/obj_cloud_vp.ply", name="Obj cloud")
    show_point_cloud("/home/ws/data/images/predicted_grasps.ply", name="Predicted grasps")
    show_point_cloud("/home/ws/data/images/filtered_grasps.ply", name="Filtered grasps")
    show_point_cloud("/home/ws/data/images/final_grasp.ply", name="Final grasps")


def main() -> None:
    # docker run -p 5001:5001 --gpus all -it craiden/openmask:v1.0 python3 app.py
    merge_clouds()
    align_clouds()
    get_mask()
    show_clouds()
    show_object_and_grasps()

if __name__ == "__main__":
    main()
