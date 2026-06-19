#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sam3_inference.srv import InferenceSam3
from sam3_inference.msg import Detection, BBox
from cv_bridge import CvBridge

from utils.sam3_infer import Sam3Inference
import torch

class Sam3Server(Node):

    def __init__(self):
        super().__init__('sam3_inference_server')

        self.bridge = CvBridge()

        self.sam3 = Sam3Inference(confidence_threshold=0.6)
        
        self.srv = self.create_service(
            InferenceSam3,
            'run_sam3_inference',
            self.handle_request
        )

        self.get_logger().info("SAM3 Service Ready")

    def handle_request(self, request, response):

        image = self.bridge.imgmsg_to_cv2(
            request.image,
            desired_encoding='rgb8'
        )
        prompts = request.prompts

        with torch.no_grad():
                detections = self.sam3.infer(
                    image,
                    prompts=prompts,
                    input_format="rgb",
                    visualize=True
                )

        print(f"Detections: {detections}")
        for det in detections:
            msg = Detection()
            msg.name = det.name
            msg.conf = float(det.conf)

            bbox = BBox()
            bbox.xmin = float(det.bbox.xmin)
            bbox.ymin = float(det.bbox.ymin)
            bbox.xmax = float(det.bbox.xmax)
            bbox.ymax = float(det.bbox.ymax)

            msg.bbox = bbox
            response.detections.append(msg)

        return response

    def run_sam3(self, image):
        # Replace with real SAM3 inference
        return [
            {
                "name": "handle",
                "conf": 0.7419861,
                "xmin": 135.77,
                "ymin": 272.88,
                "xmax": 158.62,
                "ymax": 314.62
            }
        ]


def main():
    rclpy.init()
    node = Sam3Server()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
