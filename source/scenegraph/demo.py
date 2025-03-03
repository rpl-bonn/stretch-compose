import numpy as np
import pandas as pd
from scene_graph import SceneGraph
from preprocessing import preprocess_scan
from drawer_integration import parse_txt
import open3d as o3d


if __name__ == "__main__":
    SCAN_DIR = "/home/cvg-robotics/tim_ws/spot-compose-tim/data/prescans/24-08-17a/"
    SAVE_DIR = "/home/cvg-robotics/tim_ws/3D-Scene-Understanding"

    # instantiate the label mapping for Mask3D object classes (would change if using different 3D instance segmentation model)
    label_map = pd.read_csv(SCAN_DIR + 'mask3d_label_mapping.csv', usecols=['id', 'category'])
    mask3d_label_mapping = pd.Series(label_map['category'].values, index=label_map['id']).to_dict()
    
    preprocess_scan(SCAN_DIR, drawer_detection=True, light_switch_detection=True)

    T_ipad = np.load(SCAN_DIR + "/aruco_pose.npy")

    unmovable = ["armchair", "bookshelf", "end table", "shelf", "cabinet"]
    # unmovable = []
    scene_graph = SceneGraph(label_mapping=mask3d_label_mapping, min_confidence=0.2, unmovable=unmovable, pose=T_ipad)


    drawers = False
    scene_graph.build(SCAN_DIR, drawers=drawers)

    scene_graph.color_with_ibm_palette()

    scene_graph.remove_category("curtain")

    # scene_graph.remove_node(23)

    scene_graph.visualize(labels=False, connections=True, centroids=True)


    # to transform to Spot coordinate system:
    T_spot = parse_txt("/home/cvg-robotics/tim_ws/spot-compose-tim/data/prescans/24-08-17a/icp_tform_ground.txt")
    scene_graph.change_coordinate_system(T_spot) # where T_spot is a 4x4 transformation matrix of the aruco marker in Spot coordinate system

    # to add a lamp to the scene:
    lamp_ids = []
    for idx,node in enumerate(scene_graph.nodes.values()):
        if node.sem_label == 28:
            lamp_ids.append(node.object_id)

    switch_ids = []
    for idx,node in enumerate(scene_graph.nodes.values()):
        if node.sem_label == 232:
            switch_ids.append(node.object_id)



    switch_idx_upper = switch_ids[0]
    switch_idx_lower = switch_ids[1]

    lamp_idx_upper = lamp_ids[0]
    lamp_idx_lower = lamp_ids[1]

    # e.g. add a lamp to a light switch: 
    scene_graph.nodes[switch_idx_upper].add_lamp(lamp_idx_upper)
    scene_graph.nodes[switch_idx_lower].add_lamp(lamp_idx_lower)
    #
    # scene_graph.nodes[switch_idx_upper].button_count = 1
    # scene_graph.nodes[switch_idx_upper].interaction = "PUSH"
    # scene_graph.nodes[switch_idx_lower].button_count = 1
    # scene_graph.nodes[switch_idx_lower].interaction = "PUSH"
    #
    # scene_graph.nodes[lamp_idx_upper].state = "ON"
    # scene_graph.nodes[lamp_idx_lower].state = "ON"

    scene_graph.visualize(labels=False, connections=True, centroids=True)

    # to save the point cloud as a .ply file:
    # scene_graph.save_ply(SCAN_DIR + "scene.ply")




    for node in scene_graph.nodes.values():
        pts = node.points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(f"{SAVE_DIR}/{scene_graph.label_mapping[node.sem_label]}_{node.object_id}.ply", pcd)




    a = 2