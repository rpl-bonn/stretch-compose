"""
This file contains the FrameTransformer and associated classes.
The FrameTransformer is important class to all movements, because it handles transformations between all kinds of
frames. Usually the FrameTreeSnapshot from Boston Dynamics handles frame transformations, however this class adds
some additional capabilities and is a bit more powerful.
Namely, the frame_tree_snapshot can't handle adding custom frames that well.
Additionally, the part of the frame tree we get back depends on the function we call. Typically, the different
snapshots are disjointed, leading to difficult transformations. This class serves to unify all transformations.
Finally, this class is persistent over the runtime of the program, allowing free access to all transformations at any
time. The main class here is the FrameTransformer.
"""

from __future__ import annotations

import typing

import numpy as np
from utils.coordinates import Pose2D, Pose3D

GRAPH_SEED_FRAME_NAME = "seed_graph"
VISUAL_SEED_FRAME_NAME = "seed_visual"

AnyPose = typing.Union[Pose2D, Pose3D]
Any2DPose = typing.Union[Pose2D]
Any3DPose = typing.Union[Pose3D]


def transform_pose(*inputs: AnyPose) -> AnyPose | list[AnyPose]:
    outputs = []
    for _input in inputs:
        if isinstance(_input, (Pose2D, Pose3D)):
            outputs.append(_input.as_pose())
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


class FrameTransformer:
    """
    Helper class designed to help with frame transformations of any kind.
    Persistent over runtime of program.
    """

    def __init__(self):
        self.frames_tform_vision = {}
        

    def transform(self, start_frame: str, end_frame: str, start_pose: AnyPose, in_common_pose: bool = False) -> AnyPose:
        """
        Basic transformation of a pose between two frames. Equivalent to end_tform_start
        :param start_frame: frame in which original pose is specified
        :param end_frame: frame to transform to
        :param start_pose: pose to transform (in start_frame)
        :param in_common_pose: whether inputs / outputs are given in Pose2D or Pose3D
        :return: pose in end_frame
        """
        if in_common_pose:
            start_pose = transform_pose(start_pose)

        end_tform_start = self.end_tform_start(start_frame, end_frame)
        end_pose = end_tform_start * start_pose

        if in_common_pose:
            end_pose = transform_pose(end_pose)
        return end_pose


    def transform_matrix(self, start_frame: str, end_frame: str) -> np.ndarray:
        """
        Compute transformation matrix between two frames.
        :param start_frame: frame in which pose is originally specified
        :param end_frame: frame to transform to
        :return: end_tform_start
        """
        start_pose = math_helpers.SE3Pose(0, 0, 0, math_helpers.Quat())
        end_pose = self.transform(start_frame, end_frame, start_pose)
        return end_pose.to_matrix()


    def _get_frame_tform_body(self, frame_name: str) -> math_helpers.SE3Pose:
        """
        Returns the transformation from the given frame to the BODY_FRAME_NAME frame.
        Checks normal kinematics tree, SEED_FRAME tree, and world objects tree (if given).
        """
        # first  we check whether the frame is simply the SEED_FRAME, if so we return seed_tform_body
        if frame_name == GRAPH_SEED_FRAME_NAME:
            if not self.graph_nav_client:
                raise ValueError(f"frame_name is {GRAPH_SEED_FRAME_NAME}, but graph_nav_client is not initialized!")
            seed_tform_body = (self.graph_nav_client.get_localization_state().localization.seed_tform_body)
            return math_helpers.SE3Pose.from_proto(seed_tform_body)

        if frame_name in self.frames_tform_vision:
            odom_tform_body = self._get_frame_tform_body(ODOM_FRAME_NAME)
            return self.frames_tform_vision[frame_name] * odom_tform_body

        # then we want to check whether the frame is in the basic kinematics tree
        # if yes, we can simply return that
        robot_state = self.robot_state_client.get_robot_state()
        kinematics_transforms_snapshot = robot_state.kinematic_state.transforms_snapshot
        if frame_name in list(kinematics_transforms_snapshot.child_to_parent_edge_map):
            return get_a_tform_b(kinematics_transforms_snapshot, frame_name, BODY_FRAME_NAME)

        # last we check if the frame is in any of the world objects
        if self.world_object_client.is_instantiated():
            world_objects = self.world_object_client.list_world_objects().world_objects
            for world_object in world_objects:
                # check if one of the edges in the tree contains the wanted frame
                wo_transforms = world_object.transforms_snapshot
                edges = list(wo_transforms.child_to_parent_edge_map)
                if frame_name in edges:
                    return get_a_tform_b(wo_transforms, frame_name, BODY_FRAME_NAME)

        # if neither the base kinematics, nor the world objects tree contains the frame, it does not exist
        raise ValueError(f"Frame {frame_name} could not be found as transformation!\n {kinematics_transforms_snapshot=}")


    def add_frame_tform_vision(self, name: str, tform_matrix: np.ndarray, overwrite: bool = False):
        """
        Add a new frame to the transformer. Namely, frame_tform_odom.
        :param name: name of the new frame being added
        :param tform_matrix: frame_tform_odom
        :param overwrite: assuming a frame with name "name" already exists, overwrite it?
        """
        if name in self.frames_tform_vision and not overwrite:
            raise ValueError(f"Name {name} already in frame transformer!")
        self.frames_tform_vision[name] = math_helpers.SE3Pose.from_matrix(tform_matrix)


    def end_tform_start(self, start_frame: str, end_frame: str) -> math_helpers.SE3Pose:
        """
        Calculates transformation from start_frame to end_frame, checking kinematics, seed, and world objects trees.
        """
        # The problem is that all trees are generally disjointed. But all trees contain at the least a transformation
        # to the body. Therefore, we can use this fact to combine multiple trees via the body transformation.
        start_tform_body = self._get_frame_tform_body(start_frame)
        body_tform_end = self._get_frame_tform_body(end_frame).inverse()
        start_tform_end = start_tform_body * body_tform_end
        end_tform_start = start_tform_end.inverse()
        return end_tform_start


    def get_current_body_position_in_frame(self, end_frame: str, in_common_pose: bool = False) -> Any2DPose:
        """
        Return current position of the robot in whatever frame was specified.
        :param end_frame: frame relative to which the pose should be returned
        :param in_common_pose: whether returned pose should be Pose2D / Pose3D
        """
        flat_body_center = math_helpers.SE2Pose(x=0, y=0, angle=0)
        pose = self.transform(
            start_frame=BODY_FRAME_NAME,
            end_frame=end_frame,
            start_pose=flat_body_center,
        )
        if in_common_pose:
            return Pose2D.from_bosdyn_pose(pose)
        else:
            return pose


    def get_hand_position_in_frame(self, end_frame: str, in_common_pose: bool = False) -> Any3DPose:
        """
        Return current position of the hand in whatever frame was specified.
        :param end_frame: frame relative to which the pose should be returned
        :param in_common_pose: whether returned pose should be Pose2D / Pose3D
        """
        return self.transform("hand", end_frame, Pose3D(), in_common_pose)
