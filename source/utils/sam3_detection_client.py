#!/usr/bin/env python3

import rclpy
# sys.path.append("/home/ws/ros2_ws/install/sam3_inference/lib/python3.10/site-packages")

from rclpy.node import Node
from sam3_inference.srv import InferenceSam3
from sam3_inference.msg import Detection, BBox
from cv_bridge import CvBridge
import cv2


class Sam3Client(Node):

    def __init__(self):
        super().__init__('sam3_python_client')
        self.bridge = CvBridge()
        self.get_logger().info(f"Sam3Client loaded from: {__file__}")
        self.cli = self.create_client(
            InferenceSam3,
            'run_sam3_inference'
        )
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for SAM3 service 'run_sam3_inference'...")

    def call(self, image, prompts):

        req = InferenceSam3.Request()
        req.image = self.bridge.cv2_to_imgmsg(
            image,
            encoding='bgr8'
        )
        req.prompts = prompts

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.exception() is not None:
            raise RuntimeError(f"SAM3 service call failed: {future.exception()}")

        if future.result() is None:
            raise RuntimeError("SAM3 service returned no response")

        return future.result()


def call_sam3(node, image, prompts):
    response = node.call(image, prompts)
    return response.detections

if __name__ == "__main__":
    rclpy.init()
    node = Sam3Client()
    try:
        detections = call_sam3(
            node,
            "/home/ws/data/images/gripper_cam_2.png",
            ["door", "knob"]
        )

        print("\n===== INFERENCE RESULTS =====")
        print("detections:", detections)
        for det in detections:
            print(
                f"Detection(name='{det.name}', "
                f"conf={det.conf}, "
                f"bbox=({det.bbox.xmin}, {det.bbox.ymin}, "
                f"{det.bbox.xmax}, {det.bbox.ymax}))"
            )
    finally:
        node.destroy_node()
        rclpy.shutdown()