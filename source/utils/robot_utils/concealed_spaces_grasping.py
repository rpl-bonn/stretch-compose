import numpy as np
import open3d as o3d
from utils.coordinates import Pose3D
from __future__ import annotations

import time
import numpy as np
import open3d as o3d

from utils.robot_utils.global_parameters import *
from utils.robot_utils.basic_perception import check_object_distance
from utils.robot_utils.basic_movement import *
from scipy.spatial.transform import Rotation
from utils import vis
from utils.coordinates import Pose2D, Pose3D, from_a_to_b_distanced, pose_distanced, get_door_opening_poses
from utils.graspnet_interface import predict_full_grasp
from gpd.gpd_client_api import predict_full_grasp as gpd_predict_full_grasp
from utils.importer import PointCloud
from utils.point_clouds import icp
from utils.recursive_config import Config
from utils.time import convert_time
from scripts.my_robot_scripts.graspnet_testing import visualize_grasps

from stretch_package.stretch_movement.move_body import BaseController
from stretch_package.stretch_movement.move_to_pose import JointPoseController
from stretch_package.stretch_movement.move_to_position import JointPositionController
from stretch_package.stretch_movement.stow_arm import StowArmController
from stretch_package.stretch_state.frame_transformer import FrameTransformer


def plan_drawer_top_grasp(pcd_obj: o3d.geometry.PointCloud,
                          pcd_env: o3d.geometry.PointCloud,
                          wall_margin: float = 0.03,
                          z_offset: float = 0.015) -> Pose3D | None:
    """
    Plan a top-down grasp pose inside a drawer using object and environment PCDs.
    
    :param pcd_obj: fused object point cloud
    :param pcd_env: fused environment point cloud (drawer walls, lip)
    :param wall_margin: required clearance from drawer walls
    :param z_offset: how far above the object top to start approach
    :return: Pose3D grasp pose or None if not feasible
    """
    if len(pcd_obj.points) == 0:
        print("No object points in drawer PCD.")
        return None

    pts_obj = np.asarray(pcd_obj.points)
    pts_env = np.asarray(pcd_env.points)

    # Object centroid in XY
    centroid = np.mean(pts_obj, axis=0)
    top_z = np.max(pts_obj[:, 2])

    # Candidate grasp point
    grasp_point = np.array([centroid[0], centroid[1], top_z + z_offset])

    # Environment bounding box
    min_env = np.min(pts_env, axis=0)
    max_env = np.max(pts_env, axis=0)

    # Check margins: object centroid must be safely inside
    if not (min_env[0] + wall_margin < grasp_point[0] < max_env[0] - wall_margin and
            min_env[1] + wall_margin < grasp_point[1] < max_env[1] - wall_margin):
        print("Grasp candidate too close to drawer walls.")
        return None

    # Orientation: vertical down
    rot = np.eye(3)
    rot[:, 2] = np.array([0, 0, -1])  # approach
    rot[:, 0] = np.array([1, 0, 0])   # gripper closing direction
    rot[:, 1] = np.cross(rot[:, 2], rot[:, 0])

    return Pose3D(grasp_point, rot)

from utils.robot_utils.basic_movement import carry_arm, set_gripper
from utils.robot_utils.advanced_movement import move_arm_distanced

def execute_drawer_top_grasp(pos_node, pose_node, grasp_pose: Pose3D,
                             distance_start: float = 0.12,
                             distance_end: float = 0.02,
                             grasp_width: float = 0.06):
    """
    Execute a top-down grasp in a drawer using Stretch motion primitives.
    
    :param pos_node: JointPositionController
    :param pose_node: JointPoseController
    :param grasp_pose: planned Pose3D grasp
    :param distance_start: offset before approaching (m)
    :param distance_end: offset at grasp closure (m)
    :param grasp_width: approximate object diameter
    """
    # Pre-grasp above
    move_arm_distanced(pos_node, grasp_pose, distance_start)
    set_gripper(pose_node, True)

    # Approach + close
    move_arm_distanced(pos_node, grasp_pose, distance_end)
    set_gripper(pose_node, (grasp_width - 0.06) / 0.22)

    # Retract and go to carry
    carry_arm(pose_node)
