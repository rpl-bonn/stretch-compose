import os
import time
from scripts.point_cloud_scripts import full_merge
from scripts.point_cloud_scripts import full_align
from utils.openmask_interface import get_mask_clip_features
from utils.recursive_config import Config
import open3d as o3d
from typing import Tuple


def convert_time(nanoseconds: int) -> Tuple[int, int]:
    """
    Converts nanoseconds into minutes and seconds.
    :param nanoseconds: Time in nanoseconds.
    :return: Time as tuple of minutes and seconds.
    """
    minutes = int(nanoseconds / 1e9 // 60)
    seconds = round((nanoseconds / 1e9) % 60, 2)
    return minutes, seconds


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
    config = Config()
    base_data_path = config.get_subpath("data")
    scan_path = os.path.join(str(base_data_path), "ipad_scans", f'{config["pre_scanned_graphs"]["low_res"]}', 'mesh.ply')
    autowalk_path = os.path.join(str(base_data_path), "merged_point_clouds", f'{config["pre_scanned_graphs"]["low_res"]}.ply')
    aligned_path = os.path.join(str(base_data_path), "aligned_point_clouds", f'{config["pre_scanned_graphs"]["low_res"]}', 'scene.ply')

    show_point_cloud(scan_path, name="Lidar scan")
    show_point_cloud(autowalk_path, name="Autowalk scan")
    show_point_cloud(aligned_path, name="Aligned point cloud")


def main() -> None:
    #merge_clouds()
    #align_clouds()
    # DON'T FORGET TO RUN: docker run -p 5001:5001 --gpus all -it craiden/openmask:v1.0 python3 app.py
    #get_mask()
    show_clouds()


if __name__ == "__main__":
    main()
