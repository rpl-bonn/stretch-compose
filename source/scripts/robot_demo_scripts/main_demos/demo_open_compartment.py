#!/usr/bin/env python3
import contextlib
import io
import os
import time

import cv2
import numpy as np
import rclpy
import rclpy.logging
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Transform, Twist
from rclpy.duration import Duration
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint, MultiDOFJointTrajectoryPoint

from utils.robot_utils.run_visual_servo import servo_grasp_object
from stretch_package.stretch_images.aligned_depth2color_subscriber import AlignedDepth2ColorSubscriber
from stretch_package.stretch_images.rgb_image_subscriber import RGBImageSubscriber
from utils.recursive_config import Config
from utils.robot_utils.advanced_movement import pull_and_check_drawer
from utils.robot_utils.basic_movement import get_joint_states, get_odom, set_gripper, spin_until_complete
from utils.robot_utils.basic_perception import (_handle_grasp_error_base, arm_extension_axis_base, get_arm_extension,
                                                get_depth_picture, get_rgb_picture, intrinsics_from_camera,
                                                panel_plane)
from utils.robot_utils.container_skills import IMG_DIR as GRIP_IMG_DIR, capture_gripper_view, measure_handle
from utils.robot_utils.door_open import Door, StretchArm, grip_angle, plan_stance, plan_swing, _rot, _wrap
from utils.robot_utils.door_plot import plot, save_log
from utils.robot_utils.global_parameters import MAX_ARM_POS, MAX_LIFT_POS, MIN_ARM_POS, MIN_LIFT_POS
from utils.robot_utils.robot_session import robot_session
from utils.robot_utils.visual_servo_perception import GRIPPER_CAMERA_INFO_TOPIC
from utils.sam3_detection_client import Sam3Client, call_sam3
from utils.zero_shot_object_detection_sam3 import detect_compartments_from_head, handle_shape, mask_shape

URDF_PATH = '/home/ws/configs/urdf/stretch.urdf'
KNOB_PROMPTS = ["knob", "handle", "circle", "wooden circle"]

_tracker = None
# Where the driver's odometry frame sits in the fixed frame the door is planned in; they differ
# once the odometry restarts from zero, as it can around trajectory mode
_odom_origin = (np.zeros(2), 0.0)


class HandleTracker:
    """
    Keeps the selected handle, and the other handles seen around it, in odom, so the gripper
    camera stays on that handle when others are in view.
    """

    def __init__(self, point, shape=None, neighbours=()):
        self.shape = shape
        base, theta = _odom_pose()
        to_odom = lambda p: np.array([*(base + _rot(theta, np.asarray(p, dtype=float)[:2])), float(p[2])])
        self.point = to_odom(point)
        self.neighbours = [to_odom(n) for n in neighbours]
        self.locked = False
        self.size = None

    def measure(self, robot, prompts=None, attempts=3):
        base, theta = _odom_pose()
        to_base = lambda p: np.array([*_rot(-theta, p[:2] - base), p[2]])
        kwargs = {} if prompts is None else {'prompts': prompts}
        handle = measure_handle(robot, frame='base_link', attempts=attempts, expected=to_base(self.point),
                                max_dist=0.15 if self.locked else np.inf,
                                drop_cut_off=not self.locked, shape=self.shape,
                                neighbours=[to_base(n) for n in self.neighbours], **kwargs)
        if handle is not None:
            h = np.asarray(handle.coordinates, dtype=float)[:3]
            point = np.array([*(base + _rot(theta, h[:2])), h[2]])
            self.neighbours = [n + point - self.point for n in self.neighbours]
            self.point = point
            self.locked = True
            self.size = getattr(handle, 'size_m', self.size)
        return handle


def _selected_tracker(selected):
    shape = None
    if selected.get('handle_mask') is not None:
        shape = mask_shape(selected['handle_mask'])
    if shape is None and selected.get('handle_bbox') is not None:
        bb = selected['handle_bbox']
        shape = handle_shape(float(bb.xmax - bb.xmin), float(bb.ymax - bb.ymin))
    print(f"[handle] the selected handle is {'a ' + shape if shape else 'of unknown shape'}"
          f"{' bar' if shape in ('horizontal', 'vertical') else ''}")
    return HandleTracker(selected['handle_3d_base'], shape, selected.get('other_handles', ()))


def _base_move(robot, cmd, settle=2.0, velocities=None):
    """Send one joint goal and wait for it; False if the driver stopped it short."""
    robot.pose.send_joint_pose(cmd, velocities)
    spin_until_complete(robot.pose)
    time.sleep(settle)
    return robot.pose.succeeded


def _measure_selected(robot, prompts=None, attempts=3):
    return _tracker.measure(robot, prompts, attempts)


def _raw_odom():
    pose = get_odom('/odom').pose.pose
    return np.array([pose.position.x, pose.position.y]), float(2 * np.arctan2(pose.orientation.z, pose.orientation.w))


def _odom_pose():
    """Base (x, y) and yaw in the fixed frame."""
    (b, t), (ob, ot) = _raw_odom(), _odom_origin
    return ob + _rot(ot, b), ot + t


def _rebase_odom(before, raw_switched, raw_after, turn):
    """
    Keep the fixed frame through a trajectory, during which the odometry may restart from zero:
    the raw pose at its start is whichever of raw_switched and zero the base's turn matches better.
    """
    global _odom_origin
    starts = [raw_switched, (np.zeros(2), 0.0)]
    start_b, start_t = min(starts, key=lambda s: abs(_wrap(raw_after[1] - s[1] - turn)))
    if start_t == 0.0 and not np.any(start_b):
        print("[odom] odometry restarted from zero when the trajectory started")
    ot = before[1] - start_t
    _odom_origin = (before[0] - _rot(ot, start_b), ot)


def _arm_state():
    """Base (x, y) and yaw in odom, wrist yaw and arm extension."""
    base, theta = _odom_pose()
    js = get_joint_states()
    ext = get_arm_extension()
    if ext is None:
        raise RuntimeError("no arm extension in /joint_states")
    return base, theta, float(js.position[list(js.name).index('joint_wrist_yaw')]), ext


def _joint_snapshot():
    joints = ('wrist_extension', 'joint_arm_l0', 'joint_wrist_yaw', 'joint_lift', 'joint_gripper_finger_left',
              'joint_left_wheel', 'joint_right_wheel')
    js = get_joint_states()
    names = list(js.name)
    effort = lambda i: float(js.effort[i]) if i < len(js.effort) else None
    return {n: (float(js.position[names.index(n)]), effort(names.index(n))) for n in joints if n in names}


def _measure_panel(robot, base, theta, handle=None):
    """
    The door panel in front of the gripper camera as a point and outward normal in odom, or None.
    Opens the gripper to clear the camera's view, so never call it holding the handle.
    """
    _, depth = capture_gripper_view(robot)
    K = intrinsics_from_camera(GRIPPER_CAMERA_INFO_TOPIC)
    T_base_cam = robot.tf.get_tf_matrix('base_link', 'gripper_camera_color_optical_frame')
    spin_until_complete(robot.tf)
    T_base_grasp = robot.tf.get_tf_matrix('base_link', 'link_grasp_center')
    spin_until_complete(robot.tf)
    grasp_depth = (np.linalg.inv(T_base_cam) @ T_base_grasp[:, 3])[2]

    z = (depth[..., 0] if depth.ndim == 3 else depth).astype(float) / 1000.0
    v, u = np.nonzero((z > grasp_depth + 0.01) & (z < grasp_depth + 0.30))
    z = z[v, u]
    cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)])
    pts = (T_base_cam @ cam)[:3].T
    if handle is not None:
        # Clear of the knob, and within this door rather than its neighbours
        dist = np.linalg.norm(pts - np.asarray(handle, dtype=float)[:3], axis=1)
        pts = pts[(dist > 0.04) & (dist < 0.18)]
    plane = panel_plane(pts, viewpoint=T_base_cam[:3, 3])
    if plane is None:
        print(f"[door] no door panel in the gripper camera's view ({len(pts)} depth points tried)")
        return None
    point, normal = plane
    return base + _rot(theta, point), _rot(theta, normal)


def drive_to_stance(robot, arm, plan, tol_m=0.03, tol_rad=np.radians(2.0), attempts=4):
    """Drive to the pre-grasp base pose of the plan; returns the pre-grasp joint pose."""
    target, target_theta = np.asarray(plan.pre_base), plan.theta[1]
    print(f"Base turns {np.degrees(np.ptp(plan.theta[1:])):.0f} deg in place over the planned "
          f"{np.degrees(plan.angles[-1]):.0f} deg swing; slack {plan.slack * 100:.1f} cm ({plan.limiter})")
    # Arm retracted and gripper stowed so nothing sweeps into the cabinet while the base turns
    _base_move(robot, {'wrist_extension': arm.ext_limits[0] + arm.RETRACTED_EXT})
    # Fingers closed first: open, they reach past the URDF meshes the stow yaw was checked on
    set_gripper(robot.pose, 0.0)
    _base_move(robot, {'joint_wrist_yaw': arm.STOW_YAW})
    for attempt in range(attempts + 1):
        base, theta = _odom_pose()
        gx, gy = _rot(-theta, target - base)
        dist, turn = float(np.hypot(gx, gy)), float(_wrap(target_theta - theta))
        print(f"Stance {dist * 100:.1f} cm and {np.degrees(turn):+.1f} deg away")
        if dist <= tol_m and abs(turn) <= tol_rad:
            break
        if attempt == attempts:
            print(f"Base not within {tol_m * 100:.0f} cm / {np.degrees(tol_rad):.0f} deg of the stance "
                  f"after {attempts} tries; the servo has to take up the rest")
        elif dist > tol_m:
            angle = float(np.arctan2(gy, gx))
            if abs(angle) > np.pi / 2:
                # Goal behind the base: back up rather than turn around
                angle, dist = float(_wrap(angle + np.pi)), -dist
            _base_move(robot, {'rotate_mobile_base': angle})
            _base_move(robot, {'translate_mobile_base': dist})
        else:
            _base_move(robot, {'rotate_mobile_base': turn})
    return {'joint_lift': plan.lift, 'joint_wrist_yaw': plan.yaw[1], 'joint_wrist_pitch': 0.0,
            'joint_wrist_roll': 0.0, 'wrist_extension': plan.pre_ext}


def move_to_pregrasp(robot, grasp_wp):
    """Lift and wrist first, then extend, so the arm never sweeps through the counter."""
    _base_move(robot, {k: v for k, v in grasp_wp.items() if k not in ('wrist_extension', 'rotate_mobile_base')})
    set_gripper(robot.pose, 0.65)
    _base_move(robot, {'wrist_extension': grasp_wp['wrist_extension']})


def refine_door(robot, door):
    """
    Re-measure the handle and the front with the gripper camera. Returns the door moved onto them
    and whether the front was measured; None if the selected handle is not in view.
    """
    handle = _measure_selected(robot, KNOB_PROMPTS)
    if handle is None:
        return None
    base, theta = _odom_pose()
    h = np.asarray(handle.coordinates, dtype=float)
    h_odom = base + _rot(theta, h[:2])
    panel = _measure_panel(robot, base, theta, handle=h)
    normal = door.normal if panel is None else panel[1]
    turn = float(np.arctan2(door.normal[0] * normal[1] - door.normal[1] * normal[0], door.normal @ normal))
    print(f"Gripper camera: handle {np.linalg.norm(h_odom - door.handle[:2]) * 100:.1f} cm and "
          f"{(h[2] - door.handle[2]) * 100:+.1f} cm in height from the head-camera estimate, front turned "
          f"{np.degrees(turn):+.1f} deg{'' if panel is not None else ' (front not measured)'}")
    return door.moved_to((*h_odom, h[2]), normal), panel is not None


def _gripper_direction_base(robot):
    """Horizontal unit vector in base_link the gripper points along."""
    T_wrist = robot.tf.get_tf_matrix('base_link', 'link_wrist_yaw')
    spin_until_complete(robot.tf)
    T_grasp = robot.tf.get_tf_matrix('base_link', 'link_grasp_center')
    spin_until_complete(robot.tf)
    d = T_grasp[:3, 3] - T_wrist[:3, 3]
    d[2] = 0.0
    return d / np.linalg.norm(d)


def _move_grasp_centre(robot, err, guarded=False):
    """
    Move the grasp centre by err (base_link): the base drives along x, the arm extends along y,
    the lift moves z. With guarded=True the base and arm stop on contact.
    """

    axis = arm_extension_axis_base(robot.tf)
    ext = get_arm_extension()
    if axis is None or ext is None:
        return False
    along_arm = float(err[:2] @ axis[:2])
    js = get_joint_states()
    lift = js.position[list(js.name).index('joint_lift')]
    target_ext = float(np.clip(ext + along_arm, MIN_ARM_POS, MAX_ARM_POS))
    if target_ext != ext + along_arm:
        print(f"[servo] arm needs {ext + along_arm:.3f} m, outside [{MIN_ARM_POS}, {MAX_ARM_POS}]; "
              f"clipped to {target_ext:.3f} m")
    drive = float(np.clip(err[0], -0.10, 0.10))

    set_gripper(robot.pose, 0.30)
    base_cmd = {'translate_mobile_base': (drive, 60.0) if guarded else drive} if abs(err[0]) > 0.005 else None
    arm_cmd = {'joint_lift': lift + float(np.clip(err[2], -0.15, 0.15)),
               'wrist_extension': (target_ext, 85.0) if guarded else target_ext}
    cmds = [base_cmd, arm_cmd]
    if guarded:
        g = _gripper_direction_base(robot)
        if abs(g[0]) > abs(g[:2] @ axis[:2]):
            # Gripper points along the base: line up with the arm first, drive in last
            cmds.reverse()
    for cmd in cmds:
        if cmd is not None:
            _base_move(robot, cmd)
    return True


def servo_to_handle(robot, standoff, prompts=KNOB_PROMPTS, tol=0.01, max_steps=5, final=False):
    """
    Put the grasp centre `standoff` short of the handle along the gripper's pointing direction.
    With final=True a single contact-guarded step is taken without re-measuring, since the camera
    cannot see the handle once the fingers are around it.
    """
    for step in range(1 if final else max_steps):
        handle = _measure_selected(robot, prompts)
        err = _handle_grasp_error_base(robot.tf, handle) if handle else None
        axis = arm_extension_axis_base(robot.tf)
        if err is None or axis is None:
            print(f"[servo] lost the handle at step {step + 1}")
            return False
        err = err - standoff * np.append(_gripper_direction_base(robot)[:2], 0.0)
        along_arm = float(err[:2] @ axis[:2])
        print(f"[servo] standoff {standoff*100:.0f} cm, step {step + 1}: base {err[0]*100:+.1f} "
              f"arm {along_arm*100:+.1f} lift {err[2]*100:+.1f} cm")
        if not final and max(abs(err[0]), abs(along_arm), abs(err[2])) < tol:
            return True

        if not _move_grasp_centre(robot, err, guarded=final):
            return False
    if final:
        return True
    print(f"[servo] not within {tol*100:.0f} cm of the {standoff*100:.0f} cm standoff after {max_steps} steps")
    return False


def approach_and_grasp(robot, prompts=KNOB_PROMPTS, standoffs=(0.20, 0.08, 0.04)):
    """Close in on the handle in steps, re-aiming at each, then reach past it and close the gripper."""
    for standoff in standoffs:
        if not servo_to_handle(robot, standoff, prompts):
            return False
    
    servo_to_handle(robot, -0.02, prompts, final=True)
    _close_on_handle(robot, prompts)
    return True


def _close_on_handle(robot, prompts=KNOB_PROMPTS):
    time.sleep(2.0)
    servo_grasp_object(pose_node=robot.pose, switch_mode=False, object_name=list(prompts),
                       detector=_handle_detector(robot, prompts))


def _handle_detector(robot, prompts):
    """Visual-servo detector that picks the SAM3 box nearest where the tracker expects the handle."""
    client, K = Sam3Client(), intrinsics_from_camera(GRIPPER_CAMERA_INFO_TOPIC)

    def detect(_, image):
        with contextlib.redirect_stdout(io.StringIO()):
            found = call_sam3(client, image, list(prompts))
        if not found:
            return None
        base, theta = _odom_pose()
        point = np.array([*_rot(-theta, _tracker.point[:2] - base), _tracker.point[2], 1.0])
        cam = robot.tf.get_tf_matrix('gripper_camera_color_optical_frame', 'base_link') @ point
        if cam[2] <= 0.0:
            best = max(found, key=lambda d: float(d.conf))
        else:
            u, v = K[0, 0] * cam[0] / cam[2] + K[0, 2], K[1, 1] * cam[1] / cam[2] + K[1, 2]
            best = min(found, key=lambda d: np.hypot((d.bbox.xmin + d.bbox.xmax) / 2 - u,
                                                     (d.bbox.ymin + d.bbox.ymax) / 2 - v))
        return int(best.bbox.xmin), int(best.bbox.ymin), int(best.bbox.xmax), int(best.bbox.ymax)

    return detect


def _switch_mode(robot, mode):
    """Switch stretch_driver to mode ('position', 'navigation', 'trajectory')."""
    name = f'/switch_to_{mode}_mode'
    client = robot.mode.create_client(Trigger, name)
    try:
        return robot.mode._call(client, name)
    finally:
        robot.mode.destroy_client(client)


def _follow_trajectory(robot, times, joints, base_theta):
    """
    Run waypoints as one splined trajectory in trajectory mode. joints maps joint names to a
    position per waypoint, base_theta is the base yaw per waypoint; waypoint 0 is the current pose.
    """
    t = np.asarray(times, dtype=float)

    def rates(x):
        # Central differences inside, at rest at either end
        v = np.zeros_like(x)
        v[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])
        return v

    goal = FollowJointTrajectory.Goal()
    names = list(joints)
    pos = {n: np.asarray(joints[n], dtype=float) for n in names}
    vel = {n: rates(pos[n]) for n in names}
    goal.trajectory.joint_names = names
    th = np.asarray(base_theta, dtype=float) - base_theta[0]
    w = rates(th)
    goal.multi_dof_trajectory.joint_names = ['position']
    for k, tk in enumerate(t):
        stamp = Duration(seconds=float(tk)).to_msg()
        goal.trajectory.points.append(JointTrajectoryPoint(
            positions=[float(pos[n][k]) for n in names], velocities=[float(vel[n][k]) for n in names],
            time_from_start=stamp))
        turn = Transform()
        turn.rotation.z, turn.rotation.w = float(np.sin(th[k] / 2)), float(np.cos(th[k] / 2))
        twist = Twist()
        twist.angular.z = float(w[k])
        goal.multi_dof_trajectory.points.append(MultiDOFJointTrajectoryPoint(
            transforms=[turn], velocities=[twist], time_from_start=stamp))

    before = _odom_pose()
    if not _switch_mode(robot, 'trajectory'):
        return False
    raw_switched = _raw_odom()
    try:
        robot.pose.action_client.wait_for_server()
        sent = robot.pose.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(robot.pose, sent, timeout_sec=5.0)
        handle = sent.result()
        if handle is None or not handle.accepted:
            print("[swing] the driver rejected the trajectory")
            return False
        done = handle.get_result_async()
        rclpy.spin_until_future_complete(robot.pose, done, timeout_sec=float(t[-1]) + 10.0)
        result = done.result()
        if result is None or result.result.error_code != 0:
            print(f"[swing] trajectory failed: {None if result is None else result.result.error_string}")
            return False
        return True
    finally:
        robot.switch_to_position_mode()
        _rebase_odom(before, raw_switched, _raw_odom(), float(th[-1]))


def _swing_continuous(robot, state, plan):
    """
    The whole swing as one fast trajectory, so a spring-loaded door keeps its momentum. If the
    plan stops short of fully open, the base turns another 30 deg to carry the door the rest of the way.
    """
    speeds = {'rotate_mobile_base': 0.3, 'wrist_extension': 0.05, 'joint_wrist_yaw': 0.3}
    _, theta0, yaw0, ext0 = state
    # Waypoint 1 of the plan is the grasp, where the robot is now
    theta = np.array([theta0, *plan.theta[2:]])
    theta = theta0 + np.concatenate([[0.0], np.cumsum(_wrap(np.diff(theta)))])
    yaw, ext = np.array([yaw0, *plan.yaw[2:]]), np.array([ext0, *plan.ext[2:]])
    if plan.angles[-1] < np.radians(100.0) - 1e-6:
        way = np.sign(theta[-1] - theta0) or -1
        theta = np.append(theta, theta[-1] + way * np.radians(30.0))
        yaw, ext = np.append(yaw, yaw[-1]), np.append(ext, ext[-1])
    seg = np.max([np.abs(np.diff(x)) / speeds[k]
                  for k, x in (('rotate_mobile_base', theta), ('joint_wrist_yaw', yaw), ('wrist_extension', ext))], axis=0)
    times = np.concatenate([[0.0], np.cumsum(np.maximum(seg, 0.2))])
    print(f"[swing] one trajectory through {len(times) - 1} waypoints in {times[-1]:.1f} s: base turns "
          f"{np.degrees(theta[-1] - theta0):+.0f} deg in all, "
          f"arm {ext0:.2f} -> {ext[-1]:.2f} m, wrist yaw {yaw0:+.2f} -> {yaw[-1]:+.2f}")
    return _follow_trajectory(robot, times, {'joint_wrist_yaw': yaw, 'wrist_extension': ext}, theta)


def swing_door(robot, arm, door, grip_tol, log_prefix, min_base_turn):
    """
    Open the grasped door, the base turning in place while the arm and wrist keep the grasp on the
    handle's arc. The swing is planned from where the robot actually is.
    """
    steps = []

    def measure(step):
        state = _arm_state()
        base, theta, yaw, ext = state
        grasp = arm.grasp_xy(base, theta, yaw, ext)
        angle = door.angle_of(grasp)
        entry = {'step': step, 'angle': angle, 'base': base, 'grasp': grasp, 'joints': _joint_snapshot(),
                 'measured': {'theta': theta, 'yaw': yaw, 'ext': ext},
                 'grip': grip_angle(door, arm.heading(theta, yaw), angle)}
        steps.append(entry)
        save_log(f"{log_prefix}.json", arm, door, first, steps, stage='swing', grip_tol=grip_tol)
        return state, angle, entry

    # The handle is where the grasp actually closed
    base, theta, yaw, ext = _arm_state()
    door = door.moved_to(arm.grasp_xy(base, theta, yaw, ext))
    first = plan_swing(arm, door, base, grip_angle(door, arm.heading(theta, yaw), 0.0), grip_tol, yaw_near=yaw,
                       min_base_turn=min_base_turn)
    if first is None:
        print("[swing] no door angle is reachable from where the grasp left the base")
        save_log(f"{log_prefix}.json", arm, door, None, stage='swing', grip_tol=grip_tol)
        return False
    target = first.angles[-1]
    print(f"[swing] opening to {np.degrees(target):.0f} deg, grip {np.degrees(first.grip[0]):+.0f} -> "
          f"{np.degrees(first.grip[1]):+.0f} deg off square, slack {first.slack * 100:.1f} cm ({first.limiter}), "
          f"tracking error {first.tracking_error * 1000:.1f} mm")

    state, angle, _ = measure(0)
    # Squeeze again: the gripper backs off when it stalls on the handle for long
    set_gripper(robot.pose, -0.35)
    _swing_continuous(robot, state, first)
    _, angle, _ = measure(1)

    opened = angle >= target - np.radians(3.0)
    print(f"Door {'opened' if opened else 'only'} to {np.degrees(angle):.0f} of {np.degrees(target):.0f} deg; "
          f"still holding the handle. Log: {log_prefix}.json, plot: "
          f"{plot(arm, door, first, steps=steps, path=f'{log_prefix}.png')}")
    set_gripper(robot.pose, 0.65)
    return opened


def _grip_tol_seen():
    """How far the grip may turn on the handle: a vertical bar turns in the fingers, a knob does not."""
    if _tracker is None or _tracker.size is None:
        print("[door] handle size not measured; taking it for a knob")
        return np.radians(10.0)
    w, h = _tracker.size
    bar = h >= 0.07 and h >= 1.5 * w
    tol = np.radians(45.0 if bar else 10.0)
    print(f"[door] handle {100 * w:.1f} x {100 * h:.1f} cm (w x h): a {'vertical bar' if bar else 'knob'}, "
          f"the grip may turn up to {np.degrees(tol):.0f} deg on it")
    return tol


def open_door_closed_loop(robot, selected):
    """
    Plan where to stand so the whole swing is reachable with the base turning in place, grasp the
    handle with gripper-camera feedback, then swing the door open. Logs and plots go to
    images/door_logs, to replay with python -m utils.robot_utils.door_plot.
    """
    global _tracker
    _tracker = _selected_tracker(selected)
    # Wrist yaw kept where the gripper clears the robot's own arm and mast by ~5 cm (URDF collision check)
    arm = StretchArm(URDF_PATH, ext_limits=(MIN_ARM_POS, MAX_ARM_POS),
                     yaw_limits=(-0.6, 2.6), lift_limits=(MIN_LIFT_POS, MAX_LIFT_POS))
    log_dir = os.path.join(Config().get_subpath("images"), "door_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_prefix = os.path.join(log_dir, time.strftime("door_%Y%m%d_%H%M%S"))
    detection = {k: selected.get(k) for k in ('handle_3d_base', 'hinge_3d_base', 'front_normal_base', 'left_vs_right',
                                              'front_measured')}
    door = Door.from_detection(selected, *_odom_pose())
    front_measured = bool(selected.get('front_measured', False))
    # Stance planned for a knob, which suits a bar too; the swing for what the handle turns out to be
    grip_tol = np.radians(10.0)
    back_off = 0.20
    min_base_turn = np.radians(30.0)
    # Plan the stance on the head-camera estimate, then again on the gripper camera's view if the
    # handle turns out more than 8 cm off
    for estimate in ('head-camera', 'gripper-camera'):
        normal_error = np.radians(5.0 if front_measured else 15.0)
        if not front_measured:
            print(f"[door] front not measured ({estimate}); planning for it to be up to "
                  f"{np.degrees(normal_error):.0f} deg off square")

        def stance(square, sign, min_turn, verbose=False):
            return plan_stance(arm, door, back_off=back_off, normal_error=normal_error,
                               grip_tol=grip_tol, base_turn_sign=sign, min_base_turn=min_turn,
                               square_grasp=square, verbose=verbose)

        def best_stance(square):
            # Base turning clockwise; by at least min_base_turn if that costs no door angle
            plan, grid = stance(square, -1, 0.0)
            if plan is not None:
                turning, turning_grid = stance(square, -1, min_base_turn)
                if turning is not None and turning_grid['open_to'] >= grid['open_to'] - 1e-6:
                    return turning, turning_grid
            return plan, grid

        # Square-on grasp unless off square opens the door further
        plan, grid = best_stance(True)
        free, free_grid = best_stance(False)
        if free is not None and (plan is None or free_grid['open_to'] > grid['open_to'] + 1e-6):
            square_to = 'not at all' if plan is None else f"to {np.degrees(grid['open_to']):.0f} deg"
            print(f"[door] grasping off square: it opens the door to {np.degrees(free_grid['open_to']):.0f} deg, "
                  f"square on {square_to}")
            plan, grid = free, free_grid
        if plan is None:
            print("[door] no stance with the base turning clockwise; trying either way")
            plan, grid = stance(False, None, 0.0, verbose=True)
        save_log(f"{log_prefix}_{estimate}.json", arm, door, plan, back_off=back_off,
                 detection=detection, stage=estimate, front_measured=front_measured, normal_error=normal_error,
                 grip_tol=grip_tol)
        print(f"Stance plot: {plot(arm, door, plan, grid, path=f'{log_prefix}_{estimate}.png')}")
        if plan is None:
            print(f"No stance from which this door's swing is reachable ({estimate} estimate).")
            return False
        print(f"[door] opening to {np.degrees(grid['open_to']):.0f} deg, grip {np.degrees(plan.grip[0]):+.0f} -> "
              f"{np.degrees(plan.grip[1]):+.0f} deg off square, worst-case slack {plan.robust_slack * 100:.1f} cm")
        move_to_pregrasp(robot, drive_to_stance(robot, arm, plan))
        if estimate == 'gripper-camera':
            break
        refined = refine_door(robot, door)
        if refined is None:
            print("The selected handle is not in the gripper camera's view from the stance.")
            return False
        offset = float(np.linalg.norm(refined[0].handle - door.handle))
        door, front_measured = refined
        if offset <= 0.08:
            break
        print(f"[door] the handle is {offset * 100:.1f} cm from the head-camera estimate (more than 8); "
              f"planning the stance again from the gripper camera's view")
    if not approach_and_grasp(robot, standoffs=(back_off, 0.08, 0.04)):
        print("Gripper-camera approach failed; not grasping from the head-camera estimate.")
        return False
    return swing_door(robot, arm, door, _grip_tol_seen(), f"{log_prefix}_swing", min_base_turn)


def open_drawer_closed_loop(robot, selected, standoff=0.75):
    """
    Drive in front of the selected drawer, re-detect its handle from the gripper camera,
    centre on it and pull.
    """
    global _tracker
    _tracker = _selected_tracker(selected)
    hx, hy, hz = [float(v) for v in selected['handle_3d_base']]

    # Stand out along the front normal, turned so the arm (-y of base_link) faces the front
    n = np.asarray(selected.get('front_normal_base', (-1.0, 0.0)), dtype=float)[:2]
    n /= np.linalg.norm(n)
    gx, gy = hx + standoff * n[0], hy + standoff * n[1]
    heading = float(np.arctan2(-n[0], n[1]))
    angle = float(np.arctan2(gy, gx))
    dist = float(np.hypot(gx, gy))
    print(f"Handle in base_link: ({hx:.2f}, {hy:.2f}, {hz:.2f}); driving {dist:.2f} m, "
          f"final heading {np.degrees(heading):.1f} deg")

    if dist > 0.05:
        _base_move(robot, {'rotate_mobile_base': angle})
        _base_move(robot, {'translate_mobile_base': dist})
    final_turn = (heading - angle + np.pi) % (2 * np.pi) - np.pi
    _base_move(robot, {'rotate_mobile_base': float(final_turn)})

    roll = 0.0 if selected.get('handle_orientation') == 'vertical' else np.pi / 2
    _base_move(robot, {'joint_lift': hz, 'joint_wrist_yaw': 0.0, 'joint_wrist_pitch': 0.0,
                       'joint_wrist_roll': 0.0})
    set_gripper(robot.pose, 0.4)

    if not servo_to_handle(robot, 0.20, prompts=None):
        print("Could not centre on the selected drawer handle.")
        return False

    ok = pull_and_check_drawer(robot.pose, robot.tf, lambda: _measure_selected(robot),
                               IMG_DIR=GRIP_IMG_DIR, correct_lift=True, grasp_roll=roll)
    print("Drawer opened." if ok else "Drawer pull failed.")
    return ok


def _iou(box1, box2):
    x_left, y_top = max(box1.xmin, box2.xmin), max(box1.ymin, box2.ymin)
    x_right, y_bottom = min(box1.xmax, box2.xmax), min(box1.ymax, box2.ymax)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
    area2 = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)
    return intersection / float(area1 + area2 - intersection)


def main():
    with robot_session() as robot:
        for name in ('stretch_driver', 'stretch_body', 'stretch_gripper', 'stretch_lift', 'stretch_wrist',
                     'stretch_wheel'):
            rclpy.logging.set_logger_level(name, rclpy.logging.LoggingSeverity.WARN)
        if not robot.switch_to_position_mode():
            print("Could not switch the base to position mode; base motion would be rejected. Aborting.")
            return

        rgb = get_rgb_picture(RGBImageSubscriber, robot.pose, '/camera/color/image_raw',
                              gripper=False, save_block=True, vis_block=False)
        depth = get_depth_picture(AlignedDepth2ColorSubscriber, robot.pose,
                                  '/camera/aligned_depth_to_color/image_raw', gripper=False,
                                  save_block=True, vis_block=False)

        print("Running SAM 3 detection...")
        raw_candidates = detect_compartments_from_head(rgb, depth, robot.tf)

        # Drop duplicates: boxes overlapping by more than 70%
        candidates = []
        for cand in raw_candidates:
            if all(_iou(cand.get('drawer_bbox'), kept.get('drawer_bbox')) <= 0.70 for kept in candidates):
                candidates.append(cand)

        print(f"Detected {len(candidates)} compartment candidates:")
        for idx, cand in enumerate(candidates):
            print(f"Candidate {idx}:")
            print(f"  Classification: {cand.get('classification')}")
            print(f"  Left vs Right: {cand.get('left_vs_right')}")
            print(f"  Drawer BBox: {cand.get('drawer_bbox')}")
            print(f"  Drawer Mask: {'Present' if cand.get('drawer_mask') is not None else 'None'}")
            print(f"  Handle Mask: {'Present' if cand.get('handle_mask') is not None else 'None'}")

        if not candidates:
            print("No compartments found!")
            return

        vis_img = rgb.copy()
        for idx, cand in enumerate(candidates):
            d_mask = cand.get('drawer_mask')
            d_bbox = cand.get('drawer_bbox')
            if d_mask is not None and np.any(d_mask > 0):
                mask = d_mask > 0
                vis_img[mask] = (vis_img[mask] * 0.6 + np.array([0, 255, 0], dtype=np.uint8) * 0.4).astype(np.uint8)
            h_mask = cand.get('handle_mask')
            if h_mask is not None and np.any(h_mask > 0):
                mask = h_mask > 0
                vis_img[mask] = (vis_img[mask] * 0.3 + np.array([0, 0, 255], dtype=np.uint8) * 0.7).astype(np.uint8)

            # left_vs_right is the hinge side; the label shows the handle side
            handle_side = {'left': 'right', 'right': 'left'}.get(cand['left_vs_right'], '')
            label = f"[{idx}] {cand['classification']} {handle_side}".strip()
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), _ = cv2.getTextSize(label, font, 0.6, 2)
            text_x = int((d_bbox.xmin + d_bbox.xmax) / 2) - text_w // 2
            text_y = int(d_bbox.ymax) + text_h // 2
            cv2.rectangle(vis_img, (text_x - 2, text_y - text_h - 2), (text_x + text_w + 2, text_y + 2), (0, 0, 0), -1)
            cv2.putText(vis_img, label, (text_x, text_y), font, 0.6, (0, 255, 0), 2)

        save_path = os.path.join(Config().get_subpath("images"), "detected_compartments.png")
        cv2.imwrite(save_path, vis_img)
        print(f"Saved visualization to {save_path}")
        choice = int(input(f"Select compartment [0-{len(candidates)-1}]: "))
        selected = candidates[choice]
        selected['other_handles'] = [c['handle_3d_base'] for c in candidates
                                     if c is not selected and c.get('handle_3d_base') is not None]

        if selected['classification'] == 'drawer':
            open_drawer_closed_loop(robot, selected)
        elif selected['classification'] == 'door':
            open_door_closed_loop(robot, selected)
        else:
            print(f"Unsupported classification: {selected['classification']}")


if __name__ == "__main__":
    main()
