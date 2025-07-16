import os
import numpy as np
import open3d as o3d
import pandas as pd
import cv2
from scene_graph import SceneGraph
from camera_transforms import pose_aria_pointcloud, pose_ipad_pointcloud, icp_alignment, transform_ipad_to_aria_pointcloud, spot_to_aria_coords
from utils import vis_detections, get_all_images, mask3d_labels, create_video, stitch_videos
from preprocessing import preprocess_scan
from scipy.spatial import cKDTree

if __name__ == "__main__":
    SCAN_DIR = "/home/tjark/Documents/growing_scene_graphs/SceneGraph-Drawer/D-Lab-Scan"
    DATA_FOLDER = "/home/tjark/Documents/growing_scene_graphs/SceneGraph-Drawer/"
    
    folder_names = [
        # "ball",
        # "clock_plant",
        # "cow_top_right_in_out",
        # "frame_bottom_left_in_out",
        # "frame_pillow_sim",
        # "frame_top_left",
        # "plant",
        # "plant_big_shelf",
        # "plant_big_shelf_top_left",
        # "water_can"
        # "cow_top_left",
        # "Easy_Ball_Both_Hands",
        # "Easy_Frame_Both_Hands",
        # "Easy_Frame_Left_Hand",
        # "Easy_Frame_Right_Hand",
        # "Easy_Pillow_Both_Hands",
        # "Easy_Plant_Right_Hand",
        # "Easy_WateringPot_Both_Hands",
        # "Medium_Clock_Plant",
        # "Easy_Plant_Both_Hands", ## hand-object-tracker failed
        # "Easy_Pillow_Left_Hand",  ## hand pose generation failed
        # "Easy_Pillow_Right_Hand", ## hand pose generation failed
        # "Easy_Plant_Left_Hand", ## hand pose generation failed
        # "Easy_WateringPot_Left_Hand", ## hand pose generation failed
        # "Easy_WateringPot_Right_Hand", ## hand pose generation failed
        # "MD_Pillow_Plant_Sim",
        # "MD_Frame_WateringPot_Sim",
        # "MD_Ball_double",
        # "drawer_top_right_cow",
        # "drawer_bottom_left_open",
        # "drawer_middle_left_open_close",
        # "drawer_top_left_open",
        # "drawer_top_left_open_2",
        # "Drawer_1",
        # "Drawer_2",
        # "Drawer_3",
    ]

    df = pd.read_csv('mask3d_label_mapping.csv', usecols=['id', 'category'])
    mask3d_label_mapping = pd.Series(df['category'].values, index=df['id']).to_dict()

    
    ### Needed for mask3d script:
    # mesh = o3d.io.read_triangle_mesh(SCAN_DIR + "/textured_output.obj") #, enable_post_processing=True)
    # point_cloud = o3d.io.read_point_cloud("/home/tjark/Documents/growing_scene_graphs/SceneGraph-Drawer/tim-d-lab2/tim-d-lab2.ply")
    
    # # Convert mesh vertices and point cloud points to numpy arrays
    # mesh_vertices = np.asarray(mesh.vertices)
    # pcd_points = np.asarray(point_cloud.points)
    # pcd_colors = np.asarray(point_cloud.colors)

    # # Build a KDTree for fast nearest neighbor search
    # kdtree = cKDTree(pcd_points)

    # # For each vertex in the mesh, find the nearest point in the point cloud
    # _, indices = kdtree.query(mesh_vertices)

    # # Assign colors to mesh vertices based on the nearest point in the point cloud
    # vertex_colors = pcd_colors[indices]
    # mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # # Save the mesh with colored vertices
    # o3d.io.write_triangle_mesh("colored_mesh.obj", mesh)

    # mesh = o3d.io.read_triangle_mesh("colored_mesh.obj")
    # print(mesh.has_vertex_colors())
    # o3d.visualization.draw_geometries([o3d.io.read_triangle_mesh("colored_mesh.obj")])

    
    # preprocess_scan(SCAN_DIR, drawer_detection=True)

    
    for name in folder_names:
        # TODO: this is going to be preprocessing_aria:
        if not os.path.exists(DATA_FOLDER + name + "/mesh_labelled.ply"):
            if os.path.exists(DATA_FOLDER + name + "/aruco_pose.npy"):
                T_aria = np.load(DATA_FOLDER + name + "/aruco_pose.npy")
            else:
                T_aria = pose_aria_pointcloud(DATA_FOLDER + name)
                np.save(DATA_FOLDER + name + "/aruco_pose.npy", T_aria)
            
            T_ipad = np.load(SCAN_DIR + "/aruco_pose.npy")

            icp_alignment(SCAN_DIR, DATA_FOLDER + name, T_init=np.dot(T_aria, np.linalg.inv(T_ipad)))
        
        T_aria = np.load(DATA_FOLDER + name + "/aruco_pose.npy")
        T_ipad = np.load(SCAN_DIR + "/aruco_pose.npy")
        scene_graph = SceneGraph(label_mapping=mask3d_label_mapping, min_confidence=0.2, unmovable=["armchair", "bookshelf", "end table", "shelf"], pose=T_aria)
        scene_graph.build(SCAN_DIR, DATA_FOLDER + name)

        scene_graph.color_with_ibm_palette()
        
        # scene_graph.visualize(labels=True, connections=False, centroids=False)

        # scene_graph.change_coordinate_system(T_ipad)

        # scene_graph.visualize(labels=True, connections=False, centroids=False)

        # scene_graph.track_changes(DATA_FOLDER + name)
        scene_graph.tracking_video(DATA_FOLDER + name, DATA_FOLDER + name + "/tracking.mp4")

        # scene_graph.visualize()
        
        # # #### Make the scene Graph more beautiful
        # scene_graph.remove_category("curtain")
        # scene_graph.remove_category("cabinet")
        # scene_graph.remove_category("book")
        # scene_graph.remove_category("doorframe")
        # scene_graph.label_correction()


