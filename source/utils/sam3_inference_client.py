#!/usr/bin/env python3

import rclpy
import sys
# sys.path.append("/home/ws/ros2_ws/install/sam3_inference/lib/python3.10/site-packages")

from rclpy.node import Node
from sam3_inference.srv import InferenceSam3
from sam3_inference.msg import Detection, BBox
from cv_bridge import CvBridge
import cv2


class Sam3InferenceClient(Node):

    def __init__(self):
        super().__init__('sam3_inference_client')
        self.bridge = CvBridge()
        self.cli = self.create_client(
            InferenceSam3,
            'run_sam3_inference'
        )

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service...")

    def call(self, image, prompts):

        req = InferenceSam3.Request()
        req.image = self.bridge.cv2_to_imgmsg(
            image,
            encoding='bgr8'
        )
        req.prompts = prompts

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        return future.result()


def call_sam3(node, image, prompts):
    response = node.call(image, prompts)
    return response.detections

if __name__ == "__main__":
    rclpy.init(args=None)
    sam3_client = Sam3InferenceClient()

    image_path = "/home/ws/data/images/gripper_cam_2.png"
    image = cv2.imread(image_path)
    
    response = call_sam3(sam3_client,
        image,
        ["door", "knob"]
    )

    print("\n===== INFERENCE RESULTS =====")
    print("response:", response)
    for det in response:
        print(
            f"Detection(name='{det.name}', "
            f"conf={det.conf}, "
            f"bbox=({det.bbox.xmin}, {det.bbox.ymin}, "
            f"{det.bbox.xmax}, {det.bbox.ymax}))"
        )
