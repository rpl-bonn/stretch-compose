import urdfpy
import os
import ikpy.urdf.utils
import numpy as np
from IPython import display
from utils.recursive_config import Config


config = Config()
URDF_DIR = config.get_subpath("stretch_description")


def modify_urdf():
    # Load original Stretch 3 robot description
    urdf_path = os.path.join(URDF_DIR, "original_urdf", "stretch.urdf",)
    original_urdf = urdfpy.URDF.load(urdf_path)
    print(f"name: {original_urdf.name}")
    print(f"num links: {len(original_urdf.links)}")
    print(f"num joints: {len(original_urdf.joints)}")

    # Remove links and joints not between base_link and link_grasp_center
    modified_urdf = original_urdf.copy()
    names_of_links_to_remove = ['link_right_wheel', 'link_left_wheel', 'caster_link', 
                                'link_gripper_finger_left', 'link_gripper_fingertip_left', 'link_gripper_finger_right', 'link_gripper_fingertip_right', 'link_aruco_fingertip_left', 'link_aruco_fingertip_right',
                                'link_head', 'link_head_pan', 'link_head_tilt', 'link_head_nav_cam',
                                'link_aruco_right_base', 'link_aruco_left_base', 'link_aruco_shoulder', 'link_aruco_top_wrist', 'link_aruco_inner_wrist', 'link_aruco_d405',
                                'camera_bottom_screw_frame', 'camera_link', 'gripper_camera_bottom_screw_frame', 'gripper_camera_link',
                                'gripper_camera_depth_frame', 'gripper_camera_depth_optical_frame', 'gripper_camera_color_frame', 'gripper_camera_color_optical_frame',
                                'gripper_camera_infra1_frame', 'gripper_camera_infra1_optical_frame', 'gripper_camera_infra2_frame', 'gripper_camera_infra2_optical_frame',
                                'camera_depth_frame', 'camera_depth_optical_frame', 'camera_infra1_frame', 'camera_infra1_optical_frame', 'camera_infra2_frame', 'camera_infra2_optical_frame', 'camera_color_frame', 'camera_color_optical_frame', 'camera_accel_frame', 'camera_accel_optical_frame', 'camera_gyro_frame', 'camera_gyro_optical_frame', 
                                'laser', 'respeaker_base', 'base_imu', 'link_wrist_quick_connect']
    links_to_remove = [l for l in modified_urdf._links if l.name in names_of_links_to_remove]
    for lr in links_to_remove:
        modified_urdf._links.remove(lr)
    names_of_joints_to_remove = ['joint_right_wheel', 'joint_left_wheel', 'caster_joint', 
                                 'joint_gripper_finger_left', 'joint_gripper_fingertip_left', 'joint_gripper_finger_right', 'joint_gripper_fingertip_right', 'joint_aruco_fingertip_left', 'joint_aruco_fingertip_right',
                                 'joint_head', 'joint_head_pan', 'joint_head_tilt', 'joint_head_nav_cam',
                                 'joint_aruco_right_base', 'joint_aruco_left_base', 'joint_aruco_shoulder', 'joint_aruco_top_wrist', 'joint_aruco_inner_wrist', 'joint_aruco_d405',
                                 'camera_joint', 'camera_link_joint', 'gripper_camera_joint', 'gripper_camera_link_joint',
                                 'gripper_camera_depth_joint', 'gripper_camera_depth_optical_joint', 'gripper_camera_color_joint', 'gripper_camera_color_optical_joint',
                                 'gripper_camera_infra1_joint', 'gripper_camera_infra1_optical_joint', 'gripper_camera_infra2_joint', 'gripper_camera_infra2_optical_joint',
                                 'camera_depth_joint', 'camera_depth_optical_joint', 'camera_infra1_joint', 'camera_infra1_optical_joint', 'camera_infra2_joint', 'camera_infra2_optical_joint', 'camera_color_joint', 'camera_color_optical_joint', 'camera_accel_joint', 'camera_accel_optical_joint', 'camera_gyro_joint', 'camera_gyro_optical_joint', 
                                 'joint_laser', 'joint_respeaker', 'joint_base_imu', 'joint_wrist_quick_connect']
    joints_to_remove = [l for l in modified_urdf._joints if l.name in names_of_joints_to_remove]
    for jr in joints_to_remove:
        modified_urdf._joints.remove(jr)
    print(f"name: {modified_urdf.name}")
    print(f"num links: {len(modified_urdf.links)}")
    print(f"num joints: {len(modified_urdf.joints)}")
    
    # Add new joint and link base_translation
    joint_base_translation = urdfpy.Joint(name='joint_base_translation',
                                        parent='base_link',
                                        child='link_base_translation',
                                        joint_type='prismatic',
                                        axis=np.array([1.0, 0.0, 0.0]),
                                        origin=np.eye(4, dtype=np.float64),
                                        limit=urdfpy.JointLimit(effort=100.0, velocity=1.0, lower=-1.0, upper=1.0))
    modified_urdf._joints.append(joint_base_translation)
    link_base_translation = urdfpy.Link(name='link_base_translation',
                                        inertial=None,
                                        visuals=None,
                                        collisions=None)
    modified_urdf._links.append(link_base_translation)
    
    # Adapt parent of joint_mast
    for j in modified_urdf._joints:
        if j.name == 'joint_mast':
            j.parent = 'link_base_translation'
    print(f"name: {modified_urdf.name}")
    print(f"num links: {len(modified_urdf.links)}")
    print(f"num joints: {len(modified_urdf.joints)}")
    
    # Save new modified urdf
    iktuturdf_path = os.path.join(URDF_DIR, "modified_urdf", "stretch.urdf")
    modified_urdf.save(iktuturdf_path)
    
    # Show urdf tree
    tree = ikpy.urdf.utils.get_urdf_tree(iktuturdf_path, "base_link")[0]
    display.display_png(tree)


def main():
    modify_urdf()
 
    
if __name__ == "__main__":  
    main()   