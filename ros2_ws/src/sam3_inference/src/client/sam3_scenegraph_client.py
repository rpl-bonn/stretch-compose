#!/usr/bin/env python3

import rclpy
import sys
# sys.path.append("/home/ws/ros2_ws/install/sam3_inference/lib/python3.10/site-packages")

from rclpy.node import Node
from sam3_inference.srv import SceneInferenceSam3
from sam3_inference.msg import Detection, BBox
from cv_bridge import CvBridge
import cv2


class Sam3SceneClient(Node):

    def __init__(self):
        super().__init__('sam3_scene_client')
        self.bridge = CvBridge()
        self.cli = self.create_client(
            SceneInferenceSam3,
            'run_sam3_scene_inference'
        )

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service...")

    def call(self, image, prompts, file):

        req = SceneInferenceSam3.Request()
        req.image = self.bridge.cv2_to_imgmsg(
            image,
            encoding='bgr8'
        )
        req.prompts = prompts
        req.file = file

        print('Sending request to SAM3 scene inference service...')
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        return future.result()


def call_scene_sam3(node, image, prompts, file):
    response = node.call(image, prompts, file)
    return response.detections

if __name__ == "__main__":

    response = call_scene_sam3(
        "/home/ws/data/images/gripper_cam_2.png",
        ["door", "knob"],
        "gripper_cam_2.png"
    )

    print("\n===== INFERENCE RESULTS =====")
    print("response:", response.detections)
    for det in response.detections:
        print(
            f"Detection(file: '{det.file}', "
            f"name='{det.name}', "
            f"conf={det.conf}, "
            f"bbox=({det.bbox.xmin}, {det.bbox.ymin}, "
            f"{det.bbox.xmax}, {det.bbox.ymax}))"
        )
