"""
All things video and imaging.
"""

from __future__ import annotations
import time

import cv2
import numpy as np
import os

from rclpy.node import Node
from stretch_package.stretch_images.aligned_depth2color_subscriber import AlignedDepth2ColorSubscriber
from stretch_package.stretch_images.compressed_image_subscriber import CompressedImageSubscriber
from stretch_package.stretch_images.rgb_image_subscriber import RGBImageSubscriber
from stretch_package.stretch_images.depth_image_subscriber import DepthImageSubscriber
from stretch_package.stretch_images.camera_info_subscriber import CameraInfoSubscriber
from stretch_package.stretch_state.frame_transformer import FrameTransformer
from stretch_package.stretch_movement.move_to_pose import JointPoseController

from utils.robot_utils.basic_movement import get_joint_states, set_gripper, spin_until_complete
from utils.importer import PointCloud, Vector3dVector 
from utils.recursive_config import Config


ALL_IMAGE_GREYSCALE_SOURCES = (
    CompressedImageSubscriber,
    RGBImageSubscriber,
)

ALL_IMAGE_GREYSCALE_TOPICS = (
    '/camera/color/image_raw/compressed',
    '/gripper_camera/color/image_rect_raw/compressed',
    '/camera/color/image_raw',
    '/gripper_camera/color/image_rect_raw',
)

ALL_DEPTH_SOURCES = (
    AlignedDepth2ColorSubscriber,
    DepthImageSubscriber,
)

ALL_DEPTH_TOPICS = (
    '/camera/aligned_depth_to_color/image_raw',
    '/gripper_camera/aligned_depth_to_color/image_raw',
    '/camera/depth/image_rect_raw',
    '/gripper_camera/depth/image_rect_raw'
)


def get_rgb_picture(source_node: Node, joint_node: JointPoseController, topic: str, gripper: bool = False, save_block: bool = False, vis_block: bool = False) -> np.ndarray:
    """
    Get rgb picture of specified image source.
    :param source_node: node from which to get the image
    :param topic: camera topic to which to subscribe
    :param gripper: whether the images is taken from the gripper camera
    :param save_block: whether to save the captured image
    :param vis_block: whether to show the captured image
    :return: image as np array
    """
    if gripper:
        set_gripper(joint_node, True)
    image_node = source_node(topic, not gripper, save_block, vis_block)
    image = image_node.cv_image
    image_node.destroy_node()
    return image


def get_greyscale_picture(source_node: Node, joint_node: JointPoseController, topic: str, gripper: bool = False, save_block: bool = False, vis_block: bool = False) -> np.ndarray:
    """
    Get greyscale picture of specified image source.
    :param source_node: node from which to get the image
    :param topic: camera topic to which to subscribe
    :param gripper: whether the images is taken from the gripper camera
    :param save_block: whether to save the captured image
    :param vis_block: whether to show the captured image
    :return: image as np array
    """
    if gripper:
        set_gripper(joint_node, True)
    image_node = source_node(topic, not gripper, save_block, vis_block)
    image = cv2.cvtColor(image_node.cv_image, cv2.COLOR_BGR2GRAY)
    image_node.destroy_node()
    return image


def get_depth_picture(source_node: Node, pose_node: JointPoseController, topic: str, gripper: bool = False, save_block: bool = False, vis_block: bool = False) -> np.ndarray:
    """
    Get depth picture of specified image source.
    :param source_node: node from which to get the image
    :param topic: camera topic to which to subscribe
    :param gripper: whether the images is taken from the gripper
    :param save_block: whether to save the captured image
    :param vis_block: whether to show the captured image
    :return: image as np array
    """
    if gripper:
        set_gripper(pose_node, True)
    image_node = source_node(topic, not gripper, save_block, vis_block)
    image = image_node.cv_image
    image_node.destroy_node()
    return image


def get_camera_rgbd(rgb_node: Node, depth_node: Node, rgb_topic: str, depth_topic: str, joint_node: JointPoseController, gripper: bool = False, save_block: bool = False, vis_block: bool = False) -> np.ndarray:
    """
    Capture rgbd image from specified image source.
    :param rgb_topic: sensor node from which rgb readings should be taken
    :param depth_topic: sensor node from which depth readings should be taken
    :param gripper: whether the images is taken from the gripper
    :param save_block: whether to save the rgbd image
    :param vis_block: whether to visualize the rgbd image
    :return: image as np array
    """
    if gripper:
        set_gripper(joint_node, True)
    # depth first
    depth_image = get_depth_picture(depth_node, depth_topic, gripper, save_block, vis_block)
    # color next
    color_image = get_rgb_picture(rgb_node, rgb_topic, gripper, save_block, vis_block)
    image = depth_image + color_image
    return image


def intrinsics_from_camera(topic: str) -> np.ndarray:
    """
    Extract camera intrinsics from image source.
    :return: (3, 3) np array of the camera intrinsics
    """
    image_node = CameraInfoSubscriber(topic)
    # [fx,  0, cx,
    #   0, fy, cy,
    #   0,  0,  1]
    spin_until_complete(image_node)
    info = image_node.info
    image_node.destroy_node()
    return np.asarray([
        [info.k[0], 0, info.k[2]],
        [0, info.k[4], info.k[5]],
        [0, 0, 1],
    ])
 
    
def check_object_distance(pose_node):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    rgb_img = get_rgb_picture(RGBImageSubscriber, pose_node, "/gripper_camera/color/image_rect_raw", gripper=True, save_block=True)
    d_img = get_depth_picture(AlignedDepth2ColorSubscriber, pose_node, "/gripper_camera/aligned_depth_to_color/image_raw", gripper=True, save_block=True)
    corners, ids, _ = detector.detectMarkers(rgb_img)
    # markers with id 200 and 201
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id == 200:
            pt_left = corners[i][0][2].astype(int)
        elif marker_id == 201:
            pt_right = corners[i][0][1].astype(int)
    left_depth = d_img[pt_left[1], pt_left[0]] / 1000.0
    right_depth = d_img[pt_right[1], pt_right[0]] / 1000.0
    
    # iterate over all pixels between the two points left and right and take the smallest depth > 0
    center_depth = 1.0
    i = (pt_left[1] + pt_right[1]) // 2
    for j in range(pt_left[0]+100, pt_right[0]-100):
        if d_img[i, j] / 1000.0 < center_depth and d_img[i, j] / 1000.0 > 0.0:
            center_depth = d_img[i, j] / 1000.0
    print(f"Marker depth: {(left_depth + right_depth) / 2.0}")
    print(f"Center depth: {center_depth}")
    
    aruco_center_dist = (center_depth - 0.025) - (left_depth + right_depth) / 2.0
    if aruco_center_dist > 0.0:
        return -aruco_center_dist
    return 0.0
    

def depth_image_to_point_cloud(depth_image, topic):
    """Converts a depth image into a point cloud using the camera intrinsics. 
    The point cloud is represented as a numpy array of (x,y,z) values.  
    A (min_dist * depth_scale) value that casts to an integer value <=0 will be assigned a value of 1. 
    Similarly, a (max_dist * depth_scale) value that casts to >= 2^16 will be assigned a value of 2^16 - 1.

    Args:
        depth_image: A depth image.
        min_dist (double): All points in the returned point cloud will be greater than min_dist from the image plane [meters].
        max_dist (double): All points in the returned point cloud will be less than max_dist from the image plane [meters].

    Returns:
        A numpy stack of (x,y,z) values representing depth image as a point cloud expressed in the sensor frame.
    """

    source_rows, source_cols, _ = depth_image.shape
    camera_matrix = intrinsics_from_camera(topic)
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # Compute the valid data.
    rows, cols = np.mgrid[0:source_rows, 0:source_cols]

    # Convert the valid distance data to (x,y,z) values expressed in the sensor frame.
    z = depth_image / 1000.0
    x = np.multiply(z, (cols - cx)) / fx
    y = np.multiply(z, (rows - cy)) / fy
    return np.vstack((x, y, z)).T


def point_cloud_from_camera_captures(depth_images: list[(np.ndarray)], topic: str, tf_node: FrameTransformer) -> PointCloud:
    """
    Given a list of (depth_image), compute the combined point cloud relative to the specified frame.
    :param depth_images: list of (depth_image)
    :param frame_relative_to: frame relative to which the point cloud will be returned
    :return: combined point cloud
    """
    fused_point_clouds = PointCloud()
    for depth_image in depth_images:
        pcd_camera = PointCloud()
        
        source_rows, source_cols, _ = depth_image.shape
        camera_matrix = intrinsics_from_camera(topic)
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        # Compute the valid data.
        rows, cols = np.mgrid[0:source_rows, 0:source_cols]

        # Convert the valid distance data to (x,y,z) values expressed in the sensor frame.
        z = depth_image / 1000.0
        x = (cols - cx) * z / fx
        y = (rows - cy) * z / fy
        pcd_camera.points = Vector3dVector(np.array((x, y, z, 1.0)))
        camera_tform_map = tf_node.get_tf_matrix("map", "camera_color_optical_frame")
        spin_until_complete(tf_node)
        map_tform_camera = camera_tform_map.inverse()
        pcd_map = pcd_camera @ map_tform_camera.T
        fused_point_clouds += pcd_map
    return fused_point_clouds

def _handle_grasp_error_base(transform_node, handle_pose):
    """
    (handle - grasp_center) in base_link, via kinematics only
    """

    coords_cam = np.asarray(handle_pose.coords_cam, dtype=float)
    if coords_cam[2] <= 0.05:
        return None
    
    #4x4 transform from base_link to gripper_camera
    T_base_cam = transform_node.get_tf_matrix("base_link", "gripper_camera_color_optical_frame")
    spin_until_complete(transform_node)

    #h is the handle position in base_link
    h = T_base_cam @ np.array([*coords_cam, 1.0])
    # 4x4 transform from base_link to link_grasp_center
    T_base_grasp = transform_node.get_tf_matrix("base_link", "link_grasp_center")
    spin_until_complete(transform_node)
    #return the error in base_link coordinates
    return h[:3] - T_base_grasp[:3, 3]


def correct_offsets_base_frame(transform_node, joint_pose_node, handle_pose):
    """
    Center the gripper on the handle along the base drive axis:
    translate_mobile_base by the x-component of the base_link error 
    """
    deadband_m = 0.005
    max_step_m = 0.08
    settle_s   = 2.0

    err = _handle_grasp_error_base(transform_node, handle_pose)
    if err is None:
        print("[centering] skipped: handle behind camera or no valid depth")
        return None
    
    correction_m = float(np.clip(err[0], -max_step_m, max_step_m))
    print(f"[centering] handle-grasp err(base): x={err[0]*100:+.1f} y={err[1]*100:+.1f} "
          f"z={err[2]*100:+.1f} cm -> translate_mobile_base {correction_m*100:+.1f} cm")
    if abs(correction_m) < deadband_m:
        print("[centering] within deadband, no base move")
        return err
    
    joint_pose_node.send_joint_pose({'translate_mobile_base': correction_m})
    spin_until_complete(joint_pose_node)
    time.sleep(settle_s)
    return err

def correct_vertical_offset(transform_node, joint_pose_node, handle_pose, grasp_offset_m: float = 0.0):
    """
    move joint_lift by the z-component of the base_link error from _handle_grasp_error_base.
    """

    deadband_m = 0.005
    max_step_m = 0.15
    settle_s   = 2.0

    err = _handle_grasp_error_base(transform_node, handle_pose)
    if err is None:
        print("[lift] skipped: handle behind camera or no valid depth")
        return None
    delta = float(np.clip(err[2] + grasp_offset_m, -max_step_m, max_step_m))
    joint_state = get_joint_states()
    cur_lift = joint_state.position[list(joint_state.name).index('joint_lift')]
    new_lift = cur_lift + delta

    print(f"[lift] handle-grasp err(base): z={err[2]*100:+.1f} cm  cur_lift={cur_lift*100:.1f} cm  "
          f"delta={delta*100:+.1f} cm -> new_lift={new_lift*100:.1f} cm")
    if abs(delta) < deadband_m:
        print("[lift] within deadband, no adjustment")
        return err
    
    joint_pose_node.send_joint_pose({'joint_lift': new_lift})
    spin_until_complete(joint_pose_node)
    time.sleep(settle_s)
    return err



def visualize_correction(transform_node, joint_pose_node, handle_pose, IMG_DIR):
    """
    Save a debug image showing whether the grasp will land on the handle.

    Projects two points into the current gripper-camera image:
      green cross = detected handle, red cross   = link_grasp_center
    """

    rgb_image = get_rgb_picture(
        RGBImageSubscriber, joint_pose_node,
        "/gripper_camera/color/image_rect_raw", gripper=True
    )
    camera_matrix = intrinsics_from_camera('/gripper_camera/color/camera_info')
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    def project_to_pixel(point_cam):
        """Project a camera-frame point (m) to integer pixel coords, or None if behind the camera."""
        if point_cam[2] <= 0.01:
            return None
        u = int(round(fx * point_cam[0] / point_cam[2] + cx))
        v = int(round(fy * point_cam[1] / point_cam[2] + cy))
        return u, v

    # Handle in the camera frame
    tf_cam_from_map = transform_node.get_tf_matrix(
        "gripper_camera_color_optical_frame", "map"
    )
    spin_until_complete(transform_node)
    handle_in_cam = (tf_cam_from_map @ np.array([*handle_pose.coordinates, 1.0]))[:3]

    # Grasp center in the camera frame 
    tf_cam_from_base = transform_node.get_tf_matrix(
        "gripper_camera_color_optical_frame", "base_link"
    )
    spin_until_complete(transform_node)
    tf_base_from_grasp = transform_node.get_tf_matrix("base_link", "link_grasp_center")
    spin_until_complete(transform_node)
    grasp_in_base = tf_base_from_grasp[:3, 3]
    grasp_in_cam = (tf_cam_from_base @ np.array([*grasp_in_base, 1.0]))[:3]

    # Residual error in base_link 
    handle_in_base = (np.linalg.inv(tf_cam_from_base) @ np.array([*handle_in_cam, 1.0]))[:3]
    error_in_base = handle_in_base - grasp_in_base

    handle_pixel = project_to_pixel(handle_in_cam)
    grasp_pixel = project_to_pixel(grasp_in_cam)

    #visualizing
    vis = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) if rgb_image.shape[2] == 3 else rgb_image.copy()
    if handle_pixel is not None:
        cv2.drawMarker(vis, handle_pixel, (0, 255, 0), cv2.MARKER_CROSS, 40, 2)
    if grasp_pixel is not None:
        cv2.drawMarker(vis, grasp_pixel, (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
    cv2.putText(vis,
        f"err(base): x={error_in_base[0]*100:+.1f} y={error_in_base[1]*100:+.1f} "
        f"z={error_in_base[2]*100:+.1f} cm",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(vis, "GREEN=handle  RED=grasp center",
        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    save_path = os.path.join(IMG_DIR, "centering_aim_debug.png")
    cv2.imwrite(save_path, vis)
    print(f"[centering_visualisation] aim debug image saved → {save_path}")
    print(f"[centering_visualisation] handle pixel: {handle_pixel}  grasp-center pixel: {grasp_pixel}"
          f"{'  (grasp center outside image/behind camera)' if grasp_pixel is None else ''}")
    print(f"[centering_visualisation] handle - grasp_center in base_link: "
          f"x={error_in_base[0]*100:+.1f} cm (translate_mobile_base)  "
          f"y={error_in_base[1]*100:+.1f} cm (extension axis)  "
          f"z={error_in_base[2]*100:+.1f} cm (joint_lift)")