from __future__ import annotations
import numpy as np
from utils.recursive_config import Config
import os
import sys
# sys.path.append(os.path.abspath("/home/ws/3D-Scene-Understanding"))
import pandas as pd
from scenegraph.scene_graph import SceneGraph
from scenegraph.preprocessing import preprocess_scan
from scenegraph.drawer_integration import parse_txt
import open3d as o3d

config = Config()
scan_path = config.get_subpath("ipad_scans")
graph_path = config.get_subpath("scene_graph")
aligned_pc_path = config.get_subpath("aligned_point_clouds")
ending = config["pre_scanned_graphs"]["high_res"]
DATA_DIR = config.get_subpath("data")
SCAN_DIR = os.path.join(scan_path, ending)
GRAPH_DIR = os.path.join(graph_path, ending)
TFORM_DIR = os.path.join(aligned_pc_path, ending, "pose")


def get_scene_graph(scan_path: str, tform_path: str, data_path: str) -> SceneGraph:
    label_map = pd.read_csv(data_path + '/mask3d_label_mapping.csv', usecols=['id', 'category'])
    mask3d_label_mapping = pd.Series(label_map['category'].values, index=label_map['id']).to_dict()
    preprocess_scan(scan_path)
    T_ipad = np.load(scan_path + "/aruco_pose.npy")
    scene_graph = SceneGraph(label_mapping=mask3d_label_mapping, min_confidence=0.5, immovable=[], pose=T_ipad)
    scene_graph.build(scan_path)
    scene_graph.color_with_ibm_palette()
    scene_graph.remove_categories(["door", "window", "doorframe", "radiator"])
    scene_graph.remove_categories(["book", "soap dispenser", "object"])
    # Transform to Stretch coordinate system:
    icp_tform = parse_txt(os.path.join(tform_path, "icp_tform_ground.txt")) # 4x4 transformation matrix of the aruco marker in Stretch coordinate system
    scene_graph.change_coordinate_system(icp_tform)
    return scene_graph

def main():
    scene_graph = get_scene_graph(SCAN_DIR, TFORM_DIR, DATA_DIR)
    scene_graph.save_to_json(os.path.join(GRAPH_DIR, "scene_graph.json"))
    scene_graph.visualize(labels=True, connections=True, centroids=True)
     
if __name__ == "__main__":
    main()