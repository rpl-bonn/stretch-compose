"""
Gripper-camera perception for closed-loop visual servoing
(Approach adapted from hello-robot's stretch_visual_servoing demo).
"""

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


FINGER_MARKER_IDS = {200: 'left', 201: 'right'}

# Measured sticker size
FINGER_MARKER_LENGTH_M = 0.014
ARUCO_DICTIONARY = cv2.aruco.DICT_6X6_250

# Height of the fingertip point above the mounting surface
SUCTIONCUP_HEIGHT_M = {
    'cup_top': 0.014,
    'cup_bottom': 0.007,
    'cylinder_top': 0.004,
    'cylinder_bottom': 0.0,
}

# Rigid transform from each finger's ArUco marker frame to its fingertip frame
ARUCO_TO_FINGERTIP = {
    'left': np.array([
        [-0.342020143,  0.000000000,  0.939692621,  0.014247787],
        [-0.179301805, -0.981627183, -0.065260520,  0.005896859],
        [ 0.922427821, -0.190808995,  0.335736270, -0.030786708],
        [ 0.0,          0.0,          0.0,          1.0],
    ]),
    'right': np.array([
        [ 0.342020143,  0.000000000, -0.939692621, -0.014247787],
        [-0.179301805,  0.981627183, -0.065260520,  0.005896859],
        [ 0.922427821,  0.190808995,  0.335736270, -0.030786708],
        [ 0.0,          0.0,          0.0,          1.0],
    ]),
}

# ROS aligned depth images are 16UC1 in millimetres
ROS_DEPTH_SCALE = 0.001

# Ceiling on how far behind the visible surface the grasp point may be placed
MAX_CENTER_PUSH_M = 0.04

GRIPPER_COLOR_TOPIC = '/gripper_camera/color/image_rect_raw'
GRIPPER_DEPTH_TOPIC = '/gripper_camera/aligned_depth_to_color/image_raw'
GRIPPER_CAMERA_INFO_TOPIC = '/gripper_camera/color/camera_info'


def pixel_to_3d(xy_pix, z_m: float, camera_matrix: np.ndarray) -> np.ndarray:
    """
    Deproject a pixel at a known depth into a 3D point in the camera optical frame
    """

    x_pix, y_pix = xy_pix
    f_x, f_y = camera_matrix[0, 0], camera_matrix[1, 1]
    c_x, c_y = camera_matrix[0, 2], camera_matrix[1, 2]
    return np.array([((x_pix - c_x) * z_m) / f_x, ((y_pix - c_y) * z_m) / f_y, z_m])


def pixel_from_3d(xyz, camera_matrix: np.ndarray) -> np.ndarray:
    """
    Project a 3D point in the camera optical frame back into pixel coordinates
    """

    x_in, y_in, z_in = xyz
    f_x, f_y = camera_matrix[0, 0], camera_matrix[1, 1]
    c_x, c_y = camera_matrix[0, 2], camera_matrix[1, 2]
    return np.array([((f_x * x_in) / z_in) + c_x, ((f_y * y_in) / z_in) + c_y])


def fingertip_transforms_from_urdf(urdf_path: str) -> dict:
    """
    Recompute the ArUco-to-fingertip transforms from a URDF
    """

    import urchin

    urdf = urchin.URDF.load(urdf_path, lazy_load_meshes=True)
    transforms = {}
    for side in ('left', 'right'):
        fingertip_link = urdf.link_map[f'link_gripper_fingertip_{side}']
        aruco_link = urdf.link_map[f'link_aruco_fingertip_{side}']
        fk = urdf.link_fk(links=[fingertip_link.name, aruco_link.name])
        transforms[side] = np.linalg.inv(fk[aruco_link]) @ fk[fingertip_link]
    return transforms


@dataclass
class Fingertip:
    """Pose of one fingertip in the camera optical frame."""
    side: str
    pos: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    marker_pos: np.ndarray


@dataclass
class ServoObservation:
    """
    One frame of servo perception
    """

    fingertips: dict = field(default_factory=dict)
    between_fingertips: Optional[np.ndarray] = None
    aperture_m: Optional[float] = None
    target_xyz: Optional[np.ndarray] = None
    target_width_m: Optional[float] = None
    target_surface_z_m: Optional[float] = None
    target_pixel: Optional[np.ndarray] = None

    @property
    def error(self) -> Optional[np.ndarray]:
        if self.between_fingertips is None or self.target_xyz is None:
            return None
        return self.target_xyz - self.between_fingertips

    @property
    def error_norm(self) -> Optional[float]:
        error = self.error
        return None if error is None else float(np.linalg.norm(error))

    @property
    def complete(self) -> bool:
        """True when both endpoints of the error were measured this frame."""
        return self.error is not None

    def summary(self) -> str:
        parts = [f"fingertips={sorted(self.fingertips)}"]
        if self.aperture_m is not None:
            parts.append(f"aperture={100 * self.aperture_m:.1f}cm")
        if self.target_xyz is not None:
            x, y, z = 100 * self.target_xyz
            parts.append(f"target=({x:.1f}, {y:.1f}, {z:.1f})cm")
        if self.target_width_m is not None:
            parts.append(f"width={100 * self.target_width_m:.1f}cm")
        if self.error_norm is not None:
            parts.append(f"error={100 * self.error_norm:.1f}cm")
        else:
            parts.append("error=UNAVAILABLE")
        return ' '.join(parts)


class GripperCameraServoPerception:
    """
    Turns a gripper-camera RGB-D frame into a fingertip-to-target error vector.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion_coefficients: Optional[np.ndarray] = None,
        depth_scale: float = ROS_DEPTH_SCALE,
        fingertip_part: str = 'cup_top',
        min_valid_depth_pixels: int = 20,
    ):
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        if distortion_coefficients is None:
            distortion_coefficients = np.zeros(5)
        self.distortion_coefficients = np.asarray(distortion_coefficients, dtype=np.float64).flatten()
        self.depth_scale = depth_scale
        self.height_above_mounting_surface = SUCTIONCUP_HEIGHT_M[fingertip_part]
        self.min_valid_depth_pixels = min_valid_depth_pixels

        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
        parameters = cv2.aruco.DetectorParameters()
        
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        half = FINGER_MARKER_LENGTH_M / 2.0
        
        self.marker_object_points = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ])

    @classmethod
    def from_camera_info(cls, camera_info, **kwargs) -> 'GripperCameraServoPerception':
        """
        Build from a sensor_msgs/CameraInfo message
        """
        return cls(
            camera_matrix=np.array(camera_info.k).reshape(3, 3),
            distortion_coefficients=np.array(camera_info.d),
            **kwargs,
        )

    def detect_fingertips(self, color_image: np.ndarray) -> dict:
        """
        Locate both fingertips from the finger ArUco markers
        """

        corners, ids, _ = self.detector.detectMarkers(color_image)
        fingertips = {}
        if ids is None:
            return fingertips

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            side = FINGER_MARKER_IDS.get(int(marker_id))
            if side is None:
                continue

            #compute the pose of the marker in the camera frame
            ok, rvec, tvec = cv2.solvePnP(
                objectPoints=self.marker_object_points,
                imagePoints=marker_corners.reshape(4, 2),
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.distortion_coefficients,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                continue

            marker_pos = tvec.flatten()
            marker_rotation = cv2.Rodrigues(rvec)[0]

            transform = ARUCO_TO_FINGERTIP[side]
            fingertip_rotation = marker_rotation @ transform[:3, :3]
            fingertip_pos = marker_pos + marker_rotation @ transform[:3, 3]
            fingertip_pos = fingertip_pos + self.height_above_mounting_surface * fingertip_rotation[:, 2]

            fingertips[side] = Fingertip(
                side=side,
                pos=fingertip_pos,
                x_axis=fingertip_rotation[:, 0],
                y_axis=fingertip_rotation[:, 1],
                z_axis=fingertip_rotation[:, 2],
                marker_pos=marker_pos,
            )
        return fingertips

    def locate_target(
        self,
        depth_image: np.ndarray,
        box: Optional[tuple] = None,
        mask: Optional[np.ndarray] = None,
        target_pixel: Optional[tuple] = None,
        push_to_center: bool = True,
        max_push_m: float = MAX_CENTER_PUSH_M,
    ) -> Optional[dict]:
        """
        Reduce a detection to a single 3D grasp point in the camera frame
        """

        if mask is None and box is None:
            raise ValueError("locate_target needs either a box or a mask")

        height, width = depth_image.shape[:2]
        if mask is not None:
            region = np.asarray(mask, dtype=bool)
            ys, xs = np.nonzero(region)
            if len(xs) == 0:
                return None
            min_x, max_x = int(xs.min()), int(xs.max())
            min_y, max_y = int(ys.min()), int(ys.max())
            depths = depth_image[region]
        else:
            min_x, min_y, max_x, max_y = (int(round(v)) for v in box)
            min_x, max_x = max(0, min_x), min(width - 1, max_x)
            min_y, max_y = max(0, min_y), min(height - 1, max_y)
            if max_x <= min_x or max_y <= min_y:
                return None
            depths = depth_image[min_y:max_y + 1, min_x:max_x + 1]

        #TODO: replace bbox with mask
        valid = depths[depths > 0]
        if valid.size < self.min_valid_depth_pixels:
            return None
        surface_z_m = float(np.median(valid)) * self.depth_scale

        if target_pixel is not None:
            center_pix = np.asarray(target_pixel, dtype=np.float64)
        elif mask is not None:
            center_pix = np.array([xs.mean(), ys.mean()])
        else:
            center_pix = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])

        center_xyz = pixel_to_3d(center_pix, surface_z_m, self.camera_matrix)
        left_xyz = pixel_to_3d([min_x, center_pix[1]], surface_z_m, self.camera_matrix)
        right_xyz = pixel_to_3d([max_x, center_pix[1]], surface_z_m, self.camera_matrix)
        width_m = float(np.linalg.norm(right_xyz - left_xyz))

        top_xyz = pixel_to_3d([center_pix[0], min_y], surface_z_m, self.camera_matrix)
        bottom_xyz = pixel_to_3d([center_pix[0], max_y], surface_z_m, self.camera_matrix)
        height_m = float(np.linalg.norm(bottom_xyz - top_xyz))

        target_xyz = center_xyz
        if push_to_center:
            push = min(min(width_m, height_m) / 2.0, max_push_m)
            center_ray = center_xyz / np.linalg.norm(center_xyz)
            target_xyz = center_xyz + push * center_ray

        return {
            'xyz': target_xyz,
            'width_m': width_m,
            'height_m': height_m,
            'surface_z_m': surface_z_m,
            'pixel': center_pix,
        }

    def observe(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        box: Optional[tuple] = None,
        mask: Optional[np.ndarray] = None,
        target_pixel: Optional[tuple] = None,
        push_to_center: bool = True,
    ) -> ServoObservation:
        """
        Measure fingertips and target from one synchronized RGB-D frame
        """

        observation = ServoObservation(fingertips=self.detect_fingertips(color_image))

        left = observation.fingertips.get('left')
        right = observation.fingertips.get('right')
        if left is not None and right is not None:
            observation.between_fingertips = (left.pos + right.pos) / 2.0
            observation.aperture_m = float(np.linalg.norm(left.pos - right.pos))

        if box is not None or mask is not None:
            target = self.locate_target(
                depth_image, box=box, mask=mask,
                target_pixel=target_pixel, push_to_center=push_to_center,
            )
            if target is not None:
                observation.target_xyz = target['xyz']
                observation.target_width_m = target['width_m']
                observation.target_surface_z_m = target['surface_z_m']
                observation.target_pixel = target['pixel']

        return observation

    def draw_debug(self, color_image: np.ndarray, observation: ServoObservation,
                   box: Optional[tuple] = None, axis_length_m: float = 0.02) -> np.ndarray:
        """
        Render fingertip frames, the target point and the error onto a copy of the image.
        """

        image = np.copy(color_image)

        if box is not None:
            xmin, ymin, xmax, ymax = (int(round(v)) for v in box)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (200, 200, 200), 1)

        for fingertip in observation.fingertips.values():
            origin = pixel_from_3d(fingertip.pos, self.camera_matrix)
            for axis, color in ((fingertip.x_axis, (0, 0, 255)),
                                (fingertip.y_axis, (0, 255, 0)),
                                (fingertip.z_axis, (255, 0, 0))):
                tip = pixel_from_3d(fingertip.pos + axis_length_m * axis, self.camera_matrix)
                cv2.line(image, np.round(origin).astype(int), np.round(tip).astype(int),
                         color, 1, cv2.LINE_AA)

        if observation.between_fingertips is not None:
            center = pixel_from_3d(observation.between_fingertips, self.camera_matrix)
            cv2.circle(image, np.round(center).astype(int), 3, (0, 255, 255), -1, cv2.LINE_AA)

        if observation.target_xyz is not None:
            target = pixel_from_3d(observation.target_xyz, self.camera_matrix)
            cv2.circle(image, np.round(target).astype(int), 4, (255, 0, 0), -1, cv2.LINE_AA)
            if observation.between_fingertips is not None:
                center = pixel_from_3d(observation.between_fingertips, self.camera_matrix)
                cv2.line(image, np.round(center).astype(int), np.round(target).astype(int),
                         (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image, observation.summary(), (5, image.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        return image
