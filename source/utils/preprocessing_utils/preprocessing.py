import os
import argparse
import numpy as np
import open3d as o3d
import cv2
import glob
import json

from utils.preprocessing_utils.drawer_integration import register_drawers

def pose_ipad_pointcloud(scan_dir, pcd_path=None, marker_type=cv2.aruco.DICT_APRILTAG_36h11, aruco_length=0.148, vis_detection=False):
    """ Finds the first aruco marker in the given iPad scan and returns the pose of the marker in the world frame."""
    image_files = sorted(glob.glob(os.path.join(scan_dir, 'frame_*.jpg')))

    for image_name in image_files:
        image = cv2.imread(image_name)
        ### For the first iPad scan, the image needs to be rotated 90 degree        
        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        with open(image_name[:-4] + ".json", 'r') as f:
            camera_info = json.load(f)


        cam_matrix = np.array(camera_info["intrinsics"]).reshape(3, 3)
        
        arucoDict = cv2.aruco.getPredefinedDictionary(marker_type)
        arucoParams = cv2.aruco.DetectorParameters()


        corners, ids, _ = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

        if len(corners) > 0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_length, cam_matrix, 0)
            rotation_3x3, _ = cv2.Rodrigues(rvecs)
            T_camera_marker = np.eye(4)
            T_camera_marker[:3, :3] = rotation_3x3
            T_camera_marker[:3, 3] = tvecs

            # for debugging: visualize the aruco detection
            if vis_detection:
                cv2.aruco.drawDetectedMarkers(image, corners, ids)
                cv2.drawFrameAxes(image, cam_matrix, 0, rvecs, tvecs, 0.1)
                scale = 0.4
                dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
                resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
                cv2.imshow("ipad", resized)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            T_world_camera = np.array(camera_info["cameraPoseARFrame"]).reshape(4, 4)
            
            rot_x_180 = np.array([[1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]])
            
            T_world_camera = np.dot(T_world_camera, rot_x_180)

            T_world_marker = np.dot(T_world_camera, T_camera_marker)

            if pcd_path is not None:
                # # Define the inverse tranformation I did to get pcd from y-up to z-up
                # rot_z_90 = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2), 0, 0],
                #                     [np.sin(np.pi/2), np.cos(np.pi/2), 0, 0],
                #                     [0, 0, 1, 0],
                #                     [0, 0, 0, 1]])

                # rot_x_minus_90 = np.array([[1, 0, 0, 0],
                #                         [0, np.cos(-np.pi/2), -np.sin(-np.pi/2), 0],
                #                         [0, np.sin(-np.pi/2), np.cos(-np.pi/2), 0],
                #                         [0, 0, 0, 1]])
                

                # inverse = np.dot(rot_x_minus_90, rot_z_90)

                pcd = o3d.io.read_point_cloud(pcd_path)
                # pcd.transform(inverse)

                mesh_frame_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
                
                mesh_frame_camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
                mesh_frame_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
                
                mesh_frame_camera.transform(T_world_camera)
                mesh_frame_marker.transform(T_world_marker)

                world_origin = np.array([0, 0, 0, 1])
                camera_origin = np.dot(T_world_camera, world_origin)[:3]
                marker_origin = np.dot(T_world_marker, world_origin)[:3]


                sphere_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
                sphere_marker.paint_uniform_color([1, 0, 0]) # red
                sphere_marker.translate(marker_origin)

                sphere_camera = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
                sphere_camera.paint_uniform_color([0, 1, 0]) # green
                sphere_camera.translate(camera_origin)

                o3d.visualization.draw_geometries([pcd, mesh_frame_world, mesh_frame_camera, mesh_frame_marker, sphere_camera, sphere_marker])
            
            return T_world_marker
        
    print("No marker found for pose estimation")

def preprocess_scan(scan_dir, drawer_detection=False):
    """ runs the drawer detection on the iPad scan and overwrites detected drawers in the mask3d prediction"""
    with open(scan_dir + "/predictions.txt", 'r') as file:
        lines = file.readlines()

    pcd = o3d.io.read_point_cloud(scan_dir + "/mesh_labeled.ply")
    points = np.asarray(pcd.points)

    if drawer_detection and not os.path.exists(scan_dir + "/predictions_drawers.txt"):
        next_line = len(lines)
        
        indices_drawers = register_drawers(scan_dir)
        
        drawer_lines=[]
        for indices_drawer in indices_drawers:
            binary_mask = np.zeros(points.shape[0])
            binary_mask[indices_drawer] = 1
            np.savetxt(scan_dir + f"/pred_mask/{next_line:03}.txt", binary_mask, fmt='%d')
            drawer_lines += [f"pred_mask/{next_line:03}.txt 25 1.0\n",]
            next_line += 1
        
        with open(scan_dir + "/predictions_drawers.txt", 'a') as file:
            file.writelines(drawer_lines)
    
    if not os.path.exists(scan_dir + "/aruco_pose.npy"):
        T_ipad = pose_ipad_pointcloud(scan_dir)
        np.save(scan_dir + "/aruco_pose.npy", T_ipad)

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Preprocess the iPad Scan.')
   parser.add_argument('--scan_dir', type=str, required=True, help='Path to the "all data" folder from the 3D iPad scan.')
   args = parser.parse_args()
   preprocess_scan(args.scan_dir)