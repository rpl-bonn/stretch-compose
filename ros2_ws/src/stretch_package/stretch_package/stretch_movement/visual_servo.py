"""
Closed-loop visual servoing of the gripper onto a grasp target
"""

import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

sys.path.append('/home/ws/source')
from utils.robot_utils.visual_servo_perception import (
    GRIPPER_CAMERA_INFO_TOPIC,
    GRIPPER_COLOR_TOPIC,
    GRIPPER_DEPTH_TOPIC,
    GripperCameraServoPerception,
    ServoObservation,
)


class SE3Dw3Sg3Idx:
    """qpos layout for the dex wrist 3 + SG3 gripper, mirroring hello_helpers.get_Idx"""
    ARM = 0
    LIFT = 1
    WRIST_YAW = 2
    WRIST_PITCH = 3
    WRIST_ROLL = 4
    HEAD_PAN = 5
    HEAD_TILT = 6
    GRIPPER = 7
    BASE_TRANSLATE = 8
    BASE_ROTATE = 9
    num_joints = 10


# Joints we servo
SERVO_JOINT_STATE_NAMES = {
    'lift': 'joint_lift',
    'wrist_yaw': 'joint_wrist_yaw',
    'wrist_pitch': 'joint_wrist_pitch',
    'wrist_roll': 'joint_wrist_roll',
}

ARM_SEGMENT_JOINTS = ['joint_arm_l0', 'joint_arm_l1', 'joint_arm_l2', 'joint_arm_l3']

# Proportional gains, in normalized-velocity units per meter of error. Carried over from
# hello-robot's visual_servoing_demo, where they were tuned against a 15 Hz velocity loop

DEFAULT_ERROR_GAINS = {
    'lift': 6.0,
    'arm': 6.0,
    'wrist_yaw': 4.0,
    'wrist_pitch': 6.0,
    'wrist_roll': 1.0,
}

# Ceiling on each joint's speed
DEFAULT_MAX_VELOCITIES = {
    'lift': 0.10,        # m/s
    'arm': 0.12,         # m/s
    'wrist_yaw': 0.6,    # rad/s
    'wrist_pitch': 0.6,  # rad/s
    'wrist_roll': 0.6,   # rad/s
}

# Range of motion allowed while servoing
DEFAULT_JOINT_LIMITS = {
    'lift': (0.1, 1.05),
    'arm': (0.01, 0.50),
    'wrist_yaw': (-1.0, 1.5),
    'wrist_pitch': (-1.2, 0.2),
    'wrist_roll': (-1.6, 1.6),
}

# Grasp confirmation
GRASP_EFFORT_FALLBACK_THRESHOLD = -3.0
FINGERTIPS_TOUCHING_DISTANCE = 0.038
GRASP_APERTURE_RATIOS = (0.76, 1.29)
TENNIS_BALL_APERTURE_WINDOW = (0.05, 0.085)
FULLY_OPEN_FINGERTIP_DISTANCE = 0.16

# Hard cap on how far a single control tick may move a joint, whatever the gains say
MAX_STEP_PER_TICK = {
    'lift': 0.02,        # m
    'arm': 0.02,         # m
    'wrist_yaw': 0.05,   # rad
    'wrist_pitch': 0.05, # rad
    'wrist_roll': 0.05,  # rad
}

# How far the commanded setpoint may run ahead of the measured position
MAX_SETPOINT_LEAD = {
    'lift': 0.03,        # m
    'arm': 0.03,         # m
    'wrist_yaw': 0.15,   # rad
    'wrist_pitch': 0.15, # rad
    'wrist_roll': 0.15,  # rad
}


@dataclass
class ServoResult:
    """Outcome of one servo run."""
    success: bool
    reason: str
    final_error_m: Optional[float] = None
    iterations: int = 0
    commands_sent: int = 0

    def __str__(self) -> str:
        error = 'n/a' if self.final_error_m is None else f'{100 * self.final_error_m:.2f}cm'
        return (f"ServoResult(success={self.success}, reason='{self.reason}', "
                f"error={error}, iterations={self.iterations}, commands={self.commands_sent})")


@dataclass
class GraspVerification:
    """Whether the gripper actually holds the object - measured, not assumed."""
    grasped: bool
    reason: str
    effort: Optional[float] = None
    aperture_m: Optional[float] = None

    def __str__(self) -> str:
        effort = 'n/a' if self.effort is None else f'{self.effort:.1f}'
        aperture = 'n/a' if self.aperture_m is None else f'{100 * self.aperture_m:.1f}cm'
        return (f"GraspVerification(grasped={self.grasped}, reason='{self.reason}', "
                f"effort={effort}, aperture={aperture})")


class TrackedTargetProvider:
    """
    Keeps a target box available every frame from a detector that is too slow to run every frame
    """

    def __init__(self, detector: Callable, redetect_period_s: float = 1.5,
                 search_margin_px: int = 30, min_match_score: float = 0.5):
        self.detector = detector
        self.redetect_period_s = redetect_period_s
        self.search_margin_px = search_margin_px
        self.min_match_score = min_match_score
        self.box = None
        self.template = None
        self.last_detection_time = 0.0
        self.detections = 0
        self.track_failures = 0

    def _set_box(self, color_image: np.ndarray, box) -> None:
        xmin, ymin, xmax, ymax = (int(round(v)) for v in box)
        height, width = color_image.shape[:2]
        xmin, xmax = max(0, xmin), min(width - 1, xmax)
        ymin, ymax = max(0, ymin), min(height - 1, ymax)
        if xmax - xmin < 4 or ymax - ymin < 4:
            self.box, self.template = None, None
            return
        self.box = (xmin, ymin, xmax, ymax)
        self.template = color_image[ymin:ymax + 1, xmin:xmax + 1].copy()

    def _track(self, color_image: np.ndarray):
        if self.box is None or self.template is None:
            return None
        xmin, ymin, xmax, ymax = self.box
        height, width = color_image.shape[:2]
        margin = self.search_margin_px
        sx0, sy0 = max(0, xmin - margin), max(0, ymin - margin)
        sx1, sy1 = min(width, xmax + margin + 1), min(height, ymax + margin + 1)
        search = color_image[sy0:sy1, sx0:sx1]
        if search.shape[0] < self.template.shape[0] or search.shape[1] < self.template.shape[1]:
            return None

        result = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        if score < self.min_match_score:
            self.track_failures += 1
            return None
        new_xmin, new_ymin = sx0 + location[0], sy0 + location[1]
        return (new_xmin, new_ymin,
                new_xmin + (xmax - xmin), new_ymin + (ymax - ymin))

    def __call__(self, color_image: np.ndarray):
        """
        Returns the current target box, either from the detector or the tracker.
        """

        now = time.time()
        if self.box is None or (now - self.last_detection_time) >= self.redetect_period_s:
            self.last_detection_time = now
            detected = self.detector(color_image)
            if detected is not None:
                self.detections += 1
                self._set_box(color_image, detected)
                return self.box
            # Detector missed this round, fall through and try to coast on the tracker.

        tracked = self._track(color_image)
        if tracked is not None:
            self._set_box(color_image, tracked)
        return self.box if tracked is not None else None


class VisualServoController(Node):
    """
    Drives the gripper onto a target seen by the gripper camera.
    """

    def __init__(self, error_gains: Optional[dict] = None, max_velocities: Optional[dict] = None,
                 joint_limits: Optional[dict] = None, velocity_scale: float = 1.0,
                 fingertip_part: str = 'cup_top'):
        super().__init__('visual_servo_controller_node')
        self.bridge = CvBridge()
        self.idx = SE3Dw3Sg3Idx
        self.error_gains = dict(DEFAULT_ERROR_GAINS if error_gains is None else error_gains)
        self.max_velocities = dict(DEFAULT_MAX_VELOCITIES if max_velocities is None else max_velocities)
        self.joint_limits = dict(DEFAULT_JOINT_LIMITS if joint_limits is None else joint_limits)
        self.velocity_scale = velocity_scale
        self.fingertip_part = fingertip_part

        self.perception = None
        self.color_image = None
        self.color_stamp = 0.0
        self.depth_image = None
        self.depth_stamp = 0.0
        self.joint_state = None
        self.joint_stamp = 0.0

        self.create_subscription(Image, GRIPPER_COLOR_TOPIC, self._color_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, GRIPPER_DEPTH_TOPIC, self._depth_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, GRIPPER_CAMERA_INFO_TOPIC, self._camera_info_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self._joint_state_callback, 10)
        self.create_subscription(String, '/mode', self._mode_callback, 10)
        self.mode = None

        self.command_publisher = self.create_publisher(Float64MultiArray, '/joint_pose_cmd', 10)
        self.activate_client = self.create_client(Trigger, '/activate_streaming_position')
        self.deactivate_client = self.create_client(Trigger, '/deactivate_streaming_position')

    # ---------------------------------------------------------------- callbacks

    def _color_callback(self, msg: Image) -> None:
        self.color_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.color_stamp = time.time()

    def _depth_callback(self, msg: Image) -> None:
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')
        self.depth_stamp = time.time()

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        if self.perception is None:
            self.perception = GripperCameraServoPerception.from_camera_info(
                msg, fingertip_part=self.fingertip_part)
            self.get_logger().info(
                f"Gripper camera intrinsics: fx={msg.k[0]:.1f} fy={msg.k[4]:.1f} "
                f"cx={msg.k[2]:.1f} cy={msg.k[5]:.1f} ({msg.width}x{msg.height})")

    def _joint_state_callback(self, msg: JointState) -> None:
        self.joint_state = msg
        self.joint_stamp = time.time()

    def _mode_callback(self, msg: String) -> None:
        self.mode = msg.data

    # ----------------------------------------------------------------- helpers

    def _joint_position(self, name: str) -> Optional[float]:
        if self.joint_state is None or name not in self.joint_state.name:
            return None
        return float(self.joint_state.position[self.joint_state.name.index(name)])

    def _joint_effort(self, name: str) -> Optional[float]:
        if self.joint_state is None or name not in self.joint_state.name:
            return None
        index = self.joint_state.name.index(name)
        efforts = self.joint_state.effort
        if index >= len(efforts):
            return None
        return float(efforts[index])

    def measure_aperture(self, timeout_s: float = 2.0) -> Optional[float]:
        """
        Measure the fingertip separation from the image rather than the gripper motor
        """

        deadline = time.time() + timeout_s
        while time.time() < deadline and rclpy.ok():
            self._drain_callbacks(0.05)
            if self.color_image is None or self.perception is None:
                continue
            fingertips = self.perception.detect_fingertips(self.color_image)
            left, right = fingertips.get('left'), fingertips.get('right')
            if left is not None and right is not None:
                return float(np.linalg.norm(left.pos - right.pos))
        return None

    def verify_grasp(self, expected_width_m: Optional[float] = None,
                     effort_threshold: Optional[float] = GRASP_EFFORT_FALLBACK_THRESHOLD,
                     timeout_s: float = 2.0) -> GraspVerification:
        """
        Check whether the closed gripper is actually holding the object
        """

        aperture = self.measure_aperture(timeout_s)
        effort = self._joint_effort('joint_gripper_finger_left')

        if expected_width_m:
            low = max(FINGERTIPS_TOUCHING_DISTANCE, GRASP_APERTURE_RATIOS[0] * expected_width_m)
            high = GRASP_APERTURE_RATIOS[1] * expected_width_m
        else:
            low, high = TENNIS_BALL_APERTURE_WINDOW

        squeezing = None if (effort is None or effort_threshold is None) else effort <= effort_threshold

        if aperture is None:
            
            if squeezing is None:
                return GraspVerification(False, 'aperture unreadable and no effort fallback',
                                         effort, None)
            return GraspVerification(squeezing, f"effort only, markers occluded (effort {effort:.1f} "
                                                f"vs {effort_threshold})", effort, None)

        if aperture < FINGERTIPS_TOUCHING_DISTANCE:
            return GraspVerification(False, f'fingers closed to {100 * aperture:.1f}cm, holding nothing',
                                     effort, aperture)
        if not (low <= aperture <= high):
            return GraspVerification(False, f'aperture {100 * aperture:.1f}cm outside the '
                                            f'[{100 * low:.1f}, {100 * high:.1f}]cm window for a '
                                            f'{100 * (expected_width_m or 0):.1f}cm object',
                                     effort, aperture)
        # Aperture is in the window: the fingers stopped on something object-sized
        return GraspVerification(True, f'fingers stopped at {100 * aperture:.1f}cm, an object-sized '
                                       f'gap in [{100 * low:.1f}, {100 * high:.1f}]cm',
                                 effort, aperture)

    def _arm_extension(self) -> Optional[float]:
        """
        Total arm extension in meters
        """

        extension = self._joint_position('wrist_extension')
        if extension is not None:
            return extension
        segments = [self._joint_position(name) for name in ARM_SEGMENT_JOINTS]
        if any(segment is None for segment in segments):
            return None
        return float(sum(segments))

    def _current_servo_positions(self) -> Optional[dict]:
        positions = {}
        missing = []
        for joint, state_name in SERVO_JOINT_STATE_NAMES.items():
            value = self._joint_position(state_name)
            if value is None:
                missing.append(state_name)
            else:
                positions[joint] = value

        extension = self._arm_extension()
        if extension is None:
            missing.append("wrist_extension or " + "+".join(ARM_SEGMENT_JOINTS))
        else:
            positions['arm'] = extension

        if missing:
            published = [] if self.joint_state is None else list(self.joint_state.name)
            self.get_logger().error(f"/joint_states is missing {missing}; it publishes {published}")
            return None
        return positions

    def _drain_callbacks(self, wait_s: float, max_callbacks: int = 40) -> None:
        """
        Take everything the four subscriptions have queued, so the loop acts on the newest frame
        """

        rclpy.spin_once(self, timeout_sec=wait_s)
        for _ in range(max_callbacks):
            rclpy.spin_once(self, timeout_sec=0.0)

    def _wait_for_inputs(self, timeout_s: float = 10.0) -> bool:
        """
        Spin until camera_info, images and joint states have all arrived.
        """

        deadline = time.time() + timeout_s
        while time.time() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if (self.perception is not None and self.color_image is not None
                    and self.depth_image is not None and self.joint_state is not None
                    and self.mode is not None):
                return True

        missing = [name for name, value in (('camera_info', self.perception),
                                            ('color', self.color_image),
                                            ('depth', self.depth_image),
                                            ('joint_states', self.joint_state),
                                            ('mode', self.mode))
                   if value is None]
        self.get_logger().error(f"Timed out waiting for: {', '.join(missing)}")
        return False

    def _call_trigger(self, client, name: str, timeout_s: float = 5.0) -> bool:
        if not client.wait_for_service(timeout_sec=timeout_s):
            self.get_logger().error(f"Service {name} unavailable.")
            return False
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        result = future.result()
        if result is None or not result.success:
            self.get_logger().error(f"{name} failed: {getattr(result, 'message', 'no response')}")
            return False
        return True

    def _publish_qpos(self, servo_targets: dict) -> bool:
        """
        Send one streaming position command.
        """

        head_pan = self._joint_position('joint_head_pan')
        head_tilt = self._joint_position('joint_head_tilt')
        gripper = self._joint_position('joint_gripper_finger_left')
        if head_pan is None or head_tilt is None or gripper is None:
            self.get_logger().warn("Incomplete joint state; skipping command.")
            return False

        qpos = [0.0] * self.idx.num_joints
        qpos[self.idx.ARM] = servo_targets['arm']
        qpos[self.idx.LIFT] = servo_targets['lift']
        qpos[self.idx.WRIST_YAW] = servo_targets['wrist_yaw']
        qpos[self.idx.WRIST_PITCH] = servo_targets['wrist_pitch']
        qpos[self.idx.WRIST_ROLL] = servo_targets['wrist_roll']
        qpos[self.idx.HEAD_PAN] = head_pan
        qpos[self.idx.HEAD_TILT] = head_tilt
        qpos[self.idx.GRIPPER] = gripper
        qpos[self.idx.BASE_TRANSLATE] = 0.0
        qpos[self.idx.BASE_ROTATE] = 0.0

        message = Float64MultiArray()
        message.data = [float(v) for v in qpos]
        self.command_publisher.publish(message)
        return True

    def compute_joint_targets(self, observation: ServoObservation, positions: dict,
                              dt: float, target_roll: float,
                              commanded: Optional[dict] = None) -> dict:
        """
        Turn a camera-frame error into absolute joint targets for the next tick
        with Jacobian-transpose proportional control 
        """

        x_error, y_error, _ = observation.error

        velocities = {
            'wrist_yaw': -x_error,
            'wrist_pitch': -y_error,
            'wrist_roll': target_roll - positions['wrist_roll'],
        }

        yaw = positions['wrist_yaw']
        pitch = -positions['wrist_pitch']
        roll = -positions['wrist_roll']
        rotation = Rotation.from_euler('yxz', [yaw, pitch, roll]).as_matrix()
        velocities['lift'] = float(np.dot(rotation @ np.array([0.0, -1.0, 0.0]), observation.error))
        velocities['arm'] = float(np.dot(rotation @ np.array([0.0, 0.0, 1.0]), observation.error))

        targets = {}
        for joint, raw_velocity in velocities.items():
            
            normalized = np.clip(self.error_gains[joint] * raw_velocity, -1.0, 1.0)
            step = normalized * self.max_velocities[joint] * self.velocity_scale * dt
            step = float(np.clip(step, -MAX_STEP_PER_TICK[joint], MAX_STEP_PER_TICK[joint]))

            measured = positions[joint]
            if commanded is None:
                base = measured
            else:
                lead = np.clip(commanded.get(joint, measured) - measured,
                               -MAX_SETPOINT_LEAD[joint], MAX_SETPOINT_LEAD[joint])
                base = measured + float(lead)

            low, high = self.joint_limits[joint]
            targets[joint] = float(np.clip(base + step, low, high))
        return targets

    # -------------------------------------------------------------------- main

    def servo_to_target(self, target_provider: Callable, converged_error_m: float = 0.02,
                        timeout_s: float = 20.0, rate_hz: float = 10.0,
                        max_error_m: float = 0.5, target_roll: Optional[float] = None,
                        stale_frames_before_hold: int = 5, image_age_limit_s: float = 0.5,
                        require_open_aperture_m: Optional[float] = 0.12,
                        push_to_center: bool = True,
                        dry_run: bool = False,
                        debug_image_path: Optional[str] = None) -> ServoResult:
        """
        Run the servo loop until the fingertips reach the target

        Args:
            target_provider (Callable): takes the current image, returns a box or None
            converged_error_m (float): stop once the error is below this
            timeout_s (float): give up after this long
            rate_hz (float): control rate
            max_error_m (float): refuse to servo to a target further away than this
            target_roll (float): wrist roll to hold; defaults to the roll at loop start
            stale_frames_before_hold (int): consecutive incomplete frames tolerated
            image_age_limit_s (float): reject frames older than this
            require_open_aperture_m (float): refuse to start unless the fingertips are at least
                this far apart; None skips the check
            push_to_center (bool): offset the target behind the visible surface towards the
                object's centre. Turn off for thin objects lying on a surface, where the right
                grasp point is the surface itself and any push aims under it.
            dry_run (bool): run perception and log the error every tick but never command the
                robot, and never even activate streaming position. Use this first on a new
                setup to confirm the error vector is sane before anything moves.
            debug_image_path (str): if set, write the last annotated frame here

        Returns:
            ServoResult
        """

        if not self._wait_for_inputs():
            return ServoResult(False, 'inputs unavailable')

        positions = self._current_servo_positions()
        if positions is None:
            return ServoResult(False, 'joint states incomplete')
        if target_roll is None:
            target_roll = positions['wrist_roll']

        
        if self.mode not in ('position', 'navigation'):
            return ServoResult(False, f"driver mode is '{self.mode}', needs 'position' or "
                                      f"'navigation' (see ModeController.switch_to_position_mode)")

        
        if require_open_aperture_m is not None:
            aperture = self.measure_aperture()
            if aperture is not None and aperture < require_open_aperture_m:
                return ServoResult(False, f'gripper only {100 * aperture:.1f}cm open, needs '
                                          f'{100 * require_open_aperture_m:.1f}cm before approaching')

        if not dry_run and not self._call_trigger(self.activate_client, '/activate_streaming_position'):
            return ServoResult(False, 'could not activate streaming position')

        dt = 1.0 / rate_hz
        deadline = time.time() + timeout_s
        commanded = None
        iterations = 0
        commands_sent = 0
        stale_frames = 0
        last_error = None
        last_observation = None
        result = None



        try:
            while rclpy.ok() and time.time() < deadline:
                tick_started = time.time()
                self._drain_callbacks(dt)
                iterations += 1

                now = time.time()
                if (now - self.color_stamp > image_age_limit_s
                        or now - self.depth_stamp > image_age_limit_s):
                    stale_frames += 1
                    if stale_frames > stale_frames_before_hold:
                        result = ServoResult(False, 'gripper camera stream stalled',
                                             last_error, iterations, commands_sent)
                        break
                    continue

                color_image = self.color_image
                depth_image = self.depth_image
                box = target_provider(color_image)
                observation = self.perception.observe(color_image, depth_image, box=box,
                                                      push_to_center=push_to_center)
                last_observation = (color_image, observation, box)

                if not observation.complete:
                    stale_frames += 1
                    if stale_frames > stale_frames_before_hold:
                        missing = 'target' if box is None else 'fingertip markers'
                        result = ServoResult(False, f'lost {missing}', last_error,
                                             iterations, commands_sent)
                        break
                    continue

                stale_frames = 0
                last_error = observation.error_norm

                if last_error > max_error_m:
                    result = ServoResult(False, f'target {100 * last_error:.0f}cm away, '
                                                f'beyond the {100 * max_error_m:.0f}cm servo range',
                                         last_error, iterations, commands_sent)
                    break

                if last_error <= converged_error_m:
                    result = ServoResult(True, 'converged', last_error, iterations, commands_sent)
                    break

                if dry_run:
                    error_text = ', '.join(f'{100 * v:+.1f}' for v in observation.error)
                    self.get_logger().info(f"[dry run] error {100 * last_error:.1f}cm "
                                           f"({error_text}) aperture "
                                           f"{100 * (observation.aperture_m or 0.0):.1f}cm")
                else:
                    positions = self._current_servo_positions()
                    if positions is None:
                        continue

                    targets = self.compute_joint_targets(observation, positions, dt, target_roll,
                                                         commanded=commanded)
                    if self._publish_qpos(targets):
                        commanded = targets
                        commands_sent += 1
                    if commands_sent % 10 == 1:
                        self.get_logger().info(
                            f"error {100 * last_error:.1f}cm  arm {positions['arm']:.3f}->"
                            f"{targets['arm']:.3f}  lift {positions['lift']:.3f}->"
                            f"{targets['lift']:.3f}  yaw {positions['wrist_yaw']:+.3f}"
                            f"  pitch {positions['wrist_pitch']:+.3f}")

                elapsed = time.time() - tick_started
                if elapsed < dt:
                    time.sleep(dt - elapsed)

            if result is None:
                result = ServoResult(False, 'timed out', last_error, iterations, commands_sent)
        finally:
            if not dry_run:
                self._call_trigger(self.deactivate_client, '/deactivate_streaming_position')

        if debug_image_path and last_observation is not None:
            color_image, observation, box = last_observation
            cv2.imwrite(debug_image_path,
                        self.perception.draw_debug(color_image, observation, box=box))

        self.get_logger().info(str(result))
        return result

    def servo_to_object(self, object_name: str, detector: Callable,
                        redetect_period_s: float = 1.5, **kwargs) -> ServoResult:
        """
        Servo to an object provided by a text prompt
        """
        
        provider = TrackedTargetProvider(
            detector=lambda image: detector(object_name, image),
            redetect_period_s=redetect_period_s,
        )
        result = self.servo_to_target(provider, **kwargs)
        self.get_logger().info(f"Target provider: {provider.detections} detections, "
                               f"{provider.track_failures} tracking misses.")
        return result


def main(args=None):
    """Debbugging to servo to a fixed box in the center of the image"""

    rclpy.init(args=args)
    node = VisualServoController()
    try:
        if not node._wait_for_inputs():
            return
        height, width = node.color_image.shape[:2]
        box = (width // 2 - 30, height // 2 - 30, width // 2 + 30, height // 2 + 30)
        node.get_logger().info(f"Servoing to fixed box {box} for 10 s.")
        print(node.servo_to_target(lambda _: box, timeout_s=10.0,
                                   debug_image_path='/home/ws/data/images/visual_servo_last_frame.png'))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
