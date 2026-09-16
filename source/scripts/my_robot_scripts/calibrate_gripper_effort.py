"""
Debugging script to test whether the gripper's effort reading is usable as a fallback grasp signal
"""

import argparse
import time

import rclpy

from stretch_package.stretch_movement.mode_controller import ModeController
from stretch_package.stretch_movement.move_to_pose import JointPoseController
from stretch_package.stretch_movement.visual_servo import VisualServoController
from utils.robot_utils.basic_movement import set_gripper
from utils.robot_utils.basic_perception import spin_until_complete
from utils.robot_utils.basic_movement import get_joint_states

def close_and_measure(pose_node, servo_node, width_m, settle_s=1.5):
    """
    Close the gripper the way a grasp does and read back what the robot reports.

    Args:
        pose_node (JointPoseController): node used to command the gripper
        servo_node (VisualServoController): node used to read effort and measure the aperture
        width_m (float): object width, sets how far the close is commanded
        settle_s (float): pause after closing so the effort reading stabilises

    Returns:
        tuple: (effort, aperture_m), either of which may be None
    """
    set_gripper(pose_node, max(0.0, (width_m - 0.06) / 0.22))
    time.sleep(settle_s)
    servo_node._drain_callbacks(0.2)
    return servo_node._joint_effort('joint_gripper_finger_left'), servo_node.measure_aperture()


def report(label, effort, aperture):
    effort_text = 'unreadable' if effort is None else f'{effort:+.2f}'
    aperture_text = 'unreadable' if aperture is None else f'{100 * aperture:.1f}cm'
    print(f"  {label:22} effort={effort_text:>10}   aperture={aperture_text}")

def calibrate_gripper_effort():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--width', type=float, default=0.066, help='object width in meters')
    args = parser.parse_args()

    rclpy.init(args=None)
    pose_node = JointPoseController()
    servo_node = VisualServoController()
    mode_node = ModeController()


    if not mode_node.switch_to_position_mode():
        print("Could not switch to position mode; is stretch_driver running?")
        return
    if not servo_node._wait_for_inputs():
        return

    input("\nClear the gripper of any object, then press Enter...")
    set_gripper(pose_node, True)
    time.sleep(1.0)
    empty_effort, empty_aperture = close_and_measure(pose_node, servo_node, args.width)
    report('closed on nothing:', empty_effort, empty_aperture)

    set_gripper(pose_node, True)
    input(f"\nPlace the object ({100 * args.width:.1f}cm) between the fingers, then press Enter...")
    held_effort, held_aperture = close_and_measure(pose_node, servo_node, args.width)
    report('closed on object:', held_effort, held_aperture)

    print()
    if empty_effort is None or held_effort is None:
        print("Effort is not readable, so leave the fallback disabled (effort_threshold=None).")
        return

    separation = empty_effort - held_effort
    print(f"Separation: holding is {separation:+.2f} more negative than empty.")
    if separation < 1.0:
        print("That is not enough to discriminate. Effort is unusable as a grasp signal on "
                "this robot - keep relying on the measured aperture and set the fallback to "
                "None.")
    else:
        suggested = round((empty_effort + held_effort) / 2.0, 1)
        print(f"Suggested GRASP_EFFORT_FALLBACK_THRESHOLD = {suggested} "
                f"(midway between the two).")
        print("Set it in ros2_ws/src/stretch_package/stretch_package/stretch_movement/"
                "visual_servo.py and rebuild.")

    if held_aperture is not None and empty_aperture is not None:
        print(f"\nAperture separation: {100 * (held_aperture - empty_aperture):.1f}cm "
                f"between held and empty - this is the signal verify_grasp actually uses.")




def test_retraction_after_grasp():
 
    # Lift 2 cm from wherever the arm is now
    joint_states = get_joint_states()
    joint_lift = joint_states.position[joint_states.name.index('joint_lift')]
    push_pose = {'joint_lift': joint_lift + 0.02}
    pose_node.send_joint_pose(push_pose)
    spin_until_complete(pose_node)
    time.sleep(1.0)

    push_pose = {'wrist_extension': 0.0}
    pose_node.send_joint_pose(push_pose)
    spin_until_complete(pose_node)
    time.sleep(1.0)

    push_pose = {'joint_lift': height}
    pose_node.send_joint_pose(push_pose)
    spin_until_complete(pose_node)

    time.sleep(2.0)
    # Push drawer
    push_pose = {'wrist_extension': (0.40, 50.0)}
    pose_node.send_joint_pose(push_pose)
    spin_until_complete(pose_node)
    time.sleep(10.0)
    # Open gripper before retracting to avoid snagging on the drawer frame
    # set_gripper(pose_node, 0.4)
    # Retract arm — force-limited so it stops gracefully if there is residual resistance
    push_pose = {'wrist_extension': (0.1, 40.0)}
    pose_node.send_joint_pose(push_pose)
    spin_until_complete(pose_node)
    time.sleep(10.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--width', type=float, default=0.066, help='object width in meters')
    args = parser.parse_args()

    rclpy.init(args=None)
    pose_node = JointPoseController()
    servo_node = VisualServoController()
    mode_node = ModeController()

    try:
        # calibrate_gripper_effort()
        #send lift joint pose -1 cm
        # pose_node.send_joint_pose({'joint_lift': (0.01, 100.0)})
        # spin_until_complete(pose_node)

        # gripper_pose = {'wrist_extension': 0.5}
        # pose_node.send_joint_pose(gripper_pose)
        # spin_until_complete(pose_node)

        # time.sleep(1.0)

        gripper_pose = {'joint_wrist_roll': 0.0, 'joint_wrist_pitch': 0.0, 'joint_wrist_yaw': 0.0}
        pose_node.send_joint_pose(gripper_pose)
        spin_until_complete(pose_node)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        pose_node.destroy_node()
        servo_node.destroy_node()
        mode_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
