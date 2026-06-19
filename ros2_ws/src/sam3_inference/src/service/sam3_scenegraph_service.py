#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sam3_inference.srv import SceneInferenceSam3
from sam3_inference.msg import Detection, BBox
from cv_bridge import CvBridge

from utils.sam3_scene_infer import Sam3SceneInference
import torch

class Sam3SceneServer(Node):

    def __init__(self):
        super().__init__('sam3_scene_server')

        self.bridge = CvBridge()

        self.sam3 = Sam3SceneInference(confidence_threshold=0.6)
        
        self.srv = self.create_service(
            SceneInferenceSam3,
            'run_sam3_scene_inference',
            self.handle_request
        )

        self.get_logger().info("SAM3 Scene Service Ready")

    def handle_request(self, request, response):
        print('received request: file=%s, prompts=%s' % (request.file, request.prompts))
        image = self.bridge.imgmsg_to_cv2(
            request.image,
            desired_encoding='rgb8'
        )
        prompts = request.prompts
        file = request.file

        with torch.no_grad():
                detections = self.sam3.infer(
                    image,
                    prompts=prompts,
                    file=file,
                    input_format="rgb",
                    visualize=False
                )

        print(f"Detections: {detections}")
        for det in detections:
            msg = Detection()
            msg.file = det.file
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

    def run_sam3_scene(self, image):
        # Replace with real SAM3 scene inference
        return [
            {   
                "file": "gripper_cam_2.png",
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
    node = Sam3SceneServer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
