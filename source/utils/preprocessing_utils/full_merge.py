import open3d as o3d
import glob
import os
from utils.recursive_config import Config


def load_point_clouds(path):
    combined_pc = o3d.geometry.PointCloud()
    ply_files = glob.glob(os.path.join(path, "pointcloud_*.ply"))

    for ply_file in ply_files:
        print(ply_file)
        single_pc = o3d.io.read_point_cloud(ply_file)
        combined_pc += single_pc

    return combined_pc


def save_point_cloud(combined_pc, path):
    o3d.io.write_point_cloud(path, combined_pc)


def main():
    config = Config()
    base_data_path = config.get_subpath("data")
    data_path = os.path.join(str(base_data_path), "autowalk_scans", f'{config["pre_scanned_graphs"]["low_res"]}')
    save_path = os.path.join(str(base_data_path), "merged_point_clouds", f'{config["pre_scanned_graphs"]["low_res"]}.ply')

    combined_pc = load_point_clouds(data_path)
    save_point_cloud(combined_pc, save_path)


if __name__ == "__main__":
    main()