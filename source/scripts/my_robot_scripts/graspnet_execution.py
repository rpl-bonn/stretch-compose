from stretch_package.stretch_movement.move_body import BaseController
from stretch_package.stretch_movement.move_to_pose import JointPoseController
from stretch_package.stretch_movement.move_to_position import JointPositionController
from stretch_package.stretch_movement.move_head import HeadJointController
from stretch_package.stretch_movement.stow_arm import StowArmController
from stretch_package.stretch_images.compressed_image_subscriber import CompressedImageSubscriber

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from threading import Thread
import tf2_ros

import cv2
from scripts.my_robot_scripts import graspnet_planning
from ultralytics import YOLOWorld
from robot_utils.basic_movements import *

OBJECT = 'bottle'
VIS = False

class TransformManager:
    def __init__(self, node: Node):
        self.node = node
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=100))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)
        self.node.get_logger().info("TransformListener initialized.")
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self._spin_thread = Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
        
    def _spin(self):
        self.executor.spin()
            
    def _shutdown(self):
        self.executor.shutdown()
        self._spin_thread.join()
        self.node.get_logger().info("TransformListener stopped.")


def find_object_in_image(object_name: str, vis_block=False):
    # Get image from robot camera, save it and detect object
    detected = False
    model = YOLOWorld("yolov8x-worldv2.pt")
    model.set_classes([object_name])
    image_node = CompressedImageSubscriber(vis_block)
    image_node.destroy_node()
    results = model.predict('/home/ws/data/images/head_image.png')
    results[0].show()
    image = cv2.imread(results[0].path)
    for result in results:
            for box in result.boxes:
                detected = True
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
                cls_id = int(box.cls[0])  # Class ID
                confidence = box.conf[0]  # Confidence score
                # Get class label
                label = f"{model.names[cls_id]} {confidence:.2f}"
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put label
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    image_path = '/home/ws/data/images/detection.png'
    cv2.imwrite(image_path, image)
    if vis_block:
        cv2.imshow("Object Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return detected
    

def execute_pose(args=None):
    rclpy.init(args=args)
    
    # Initialize nodes
    node = rclpy.create_node('transform_manager_node')
    transform_manager = TransformManager(node)
    stow_node = StowArmController()
    base_controller_node = BaseController()
    joint_pose_controller_node = JointPoseController()
    head_controller_node = HeadJointController(transform_manager.tf_buffer)
    joint_position_controller_node = JointPositionController(transform_manager.tf_buffer)
    
    # Get body and grasp pose
    body_pose_distanced, grasp_pose_new = graspnet_planning.find_item(OBJECT, VIS)
    print(f"Planned poses: body={body_pose_distanced}, grasp={grasp_pose_new}")

    # Stow robot 
    stow_arm(stow_node)

    # Move body to goal position
    move_body(base_controller_node, body_pose_distanced.to_dimension(2))
    
    # Turn body towards goal object
    turn_body(joint_pose_controller_node, grasp_pose_new.to_dimension(2))
    
    # Look at goal object
    gaze(head_controller_node, grasp_pose_new)
    
    # Detect object
    found = find_object_in_image(OBJECT, VIS)
    print(f"Object found: {found}")

    if found:
        # Lift arm
        unstow_arm(joint_pose_controller_node, grasp_pose_new)

        # Move arm to pre_grasp_pose
        move_arm_distanced(joint_position_controller_node, grasp_pose_new, 0.2)

        # Open gripper
        set_gripper(joint_pose_controller_node, True)

        # Move arm to goal position
        move_arm(joint_position_controller_node, grasp_pose_new)
        
        # Close gripper
        set_gripper(joint_pose_controller_node, False)

        # Carry object
        carry_arm(joint_pose_controller_node)
        
    else:
        node.get_logger().info("Object is not at its place anymore. Searching for object at different location.")

    # Destroy nodes
    transform_manager._shutdown()
    node.destroy_node()
    stow_node.destroy_node()
    base_controller_node.destroy_node()
    joint_pose_controller_node.destroy_node()
    head_controller_node.destroy_node()
    joint_position_controller_node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    execute_pose()
   
#pre_grasp_pose: Pose3D(coords=(0.76, 3.68, 1.12), direction=(0.79, 0.40, -0.46))
#grasp_pose: Pose3D(coords=(0.91, 3.94, 1.12), direction=(0.79, 0.40, -0.46))
#body_pose: Pose3D(coords=(0.61, 3.42, 1.00), direction=(0.49, 0.84, 0.23))
#pos = np.array([0.8, -0.3, 0.7])
#dir = (0.0, 0.0, 0.0)
