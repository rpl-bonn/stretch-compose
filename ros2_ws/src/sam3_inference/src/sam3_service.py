#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rosidl_generator_py import import_type_support
from sam3_inference.srv import InferenceSam3
from sam3_inference.msg import Detection, BBox
from cv_bridge import CvBridge


from utils.sam3_infer import Sam3Inference
import torch
from cv_bridge import CvBridgeError


class Sam3Server(Node):

    def __init__(self):
        super().__init__('sam3_server')

        self.bridge = CvBridge()

        self.sam3 = Sam3Inference(confidence_threshold=0.6)
        
        self.srv = self.create_service(
            InferenceSam3,
            'run_sam3_inference',
            self.handle_request
        )

        self._log_runtime_bindings()

        self.get_logger().info("SAM3 Service Ready")

    def _log_runtime_bindings(self):
        import sam3_inference
        import sam3_inference.msg._detection as det_mod

        ts = import_type_support('sam3_inference')
        self.get_logger().info(
            "sam3 runtime bindings: "
            f"pkg={sam3_inference.__file__} "
            f"det_mod={det_mod.__file__} "
            f"typesupport={getattr(ts, '__file__', None)}"
        )

    def handle_request(self, request, response):
        self.get_logger().info(
        f"sam3_service file={__file__} "
        f"encoding='{request.image.encoding}' "
        f"h={request.image.height} w={request.image.width} "
        f"step={request.image.step} data_len={len(request.image.data)} "
        f"prompts={list(request.prompts)}")
        
        try:
            response.detections = []
            img_msg = request.image

            # Guard against malformed requests (empty encoding).
            if not img_msg.encoding:
                # Infer a reasonable default from bytes-per-pixel.
                if img_msg.width > 0:
                    bpp = img_msg.step // img_msg.width
                else:
                    bpp = 0

                if bpp == 3:
                    img_msg.encoding = "bgr8"
                elif bpp == 1:
                    img_msg.encoding = "mono8"
                elif bpp == 2:
                    img_msg.encoding = "16UC1"
                else:
                    self.get_logger().error(
                        f"Cannot infer encoding: width={img_msg.width}, step={img_msg.step}, data_len={len(img_msg.data)}"
                    )
                    response.detections = []
                    return response

                self.get_logger().warn(f"Incoming image had empty encoding; inferred {img_msg.encoding}")

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

            self.get_logger().info(
                f"sam3 raw detections type={type(detections).__name__} len={len(detections)}"
            )

            ros_detections = []
            for idx, det in enumerate(detections):
                self.get_logger().info(
                    f"raw[{idx}] type={type(det).__name__} "
                    f"name={getattr(det, 'name', None)} conf={getattr(det, 'conf', None)} "
                    f"bbox_type={type(getattr(det, 'bbox', None)).__name__}"
                )

                if not hasattr(det, 'bbox'):
                    self.get_logger().warn(f"Skipping detection {idx}: missing bbox")
                    continue

                x0 = float(det.bbox.xmin)
                y0 = float(det.bbox.ymin)
                x1 = float(det.bbox.xmax)
                y1 = float(det.bbox.ymax)
                conf = float(det.conf)

                if not all(math.isfinite(v) for v in (x0, y0, x1, y1, conf)):
                    self.get_logger().warn(f"Skipping detection {idx}: non-finite values")
                    continue

                msg = Detection()
                msg.name = str(det.name)
                msg.conf = conf

                bbox = BBox()
                bbox.xmin = x0
                bbox.ymin = y0
                bbox.xmax = x1
                bbox.ymax = y1

                msg.bbox = bbox
                ros_detections.append(msg)

            response.detections = ros_detections
            self.get_logger().info(
                f"sam3 response detections len={len(response.detections)} "
                f"elem_types={[type(d).__name__ for d in response.detections]}"
            )

            return response
        
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge conversion failed: {e}")
            response.detections = []
            return response
        except Exception as e:
            self.get_logger().error(f"SAM3 request failed: {e}")
            response.detections = []
            return response

# def handle_request(self, request, response):
#     try:
#         img_msg = request.image

#         # Guard against malformed requests (empty encoding).
#         if not img_msg.encoding:
#             # Infer a reasonable default from bytes-per-pixel.
#             if img_msg.width > 0:
#                 bpp = img_msg.step // img_msg.width
#             else:
#                 bpp = 0

#             if bpp == 3:
#                 img_msg.encoding = "bgr8"
#             elif bpp == 1:
#                 img_msg.encoding = "mono8"
#             elif bpp == 2:
#                 img_msg.encoding = "16UC1"
#             else:
#                 self.get_logger().error(
#                     f"Cannot infer encoding: width={img_msg.width}, step={img_msg.step}, data_len={len(img_msg.data)}"
#                 )
#                 response.detections = []
#                 return response

#             self.get_logger().warn(f"Incoming image had empty encoding; inferred {img_msg.encoding}")

#         image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

#         # ... run inference and fill response ...
#         return response

#     except CvBridgeError as e:
#         self.get_logger().error(f"CvBridge conversion failed: {e}")
#         response.detections = []
#         return response
#     except Exception as e:
#         self.get_logger().error(f"SAM3 request failed: {e}")
#         response.detections = []
#         return response

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
