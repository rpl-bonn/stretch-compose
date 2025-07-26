"""
All things video and imaging.
"""

from __future__ import annotations

import cv2
import numpy as np

from rclpy.node import Node
from stretch_package.stretch_images.aligned_depth2color_subscriber import AlignedDepth2ColorSubscriber
from stretch_package.stretch_images.compressed_image_subscriber import CompressedImageSubscriber
from stretch_package.stretch_images.rgb_image_subscriber import RGBImageSubscriber
from stretch_package.stretch_images.depth_image_subscriber import DepthImageSubscriber
from stretch_package.stretch_images.camera_info_subscriber import CameraInfoSubscriber
from stretch_package.stretch_state.frame_transformer import FrameTransformer
from stretch_package.stretch_movement.move_to_pose import JointPoseController

from utils.robot_utils.basic_movement import set_gripper, spin_until_complete
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