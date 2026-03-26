#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class DetectionVisualizer:
    def __init__(self, topic="annotated_image"):
        self._owns_context = False
        if not rclpy.ok():
            rclpy.init()
            self._owns_context = True
        self._node = Node("detection_visualizer")
        self._bridge = CvBridge()
        self._pub = self._node.create_publisher(Image, topic, 10)

    def visualize(self, img_path: str,
                  detections: list[dict] = None,
                  masks: np.ndarray = None,
                  scores: np.ndarray = None,
                  logits: np.ndarray = None,
                  point_coords: np.ndarray = None,
                  input_labels: np.ndarray = None,
                  input_box: np.ndarray = None):
        """
        Visualize YOLO detections and/or SAM masks on an image loaded from img_path.

        Args:
            img_path (str): Path to the image file.
            detections (list[dict], optional): [{'label': str, 'confidence': float, 'box': [x1,y1,x2,y2]}].
            masks (np.ndarray, optional): SAM binary masks.
            scores (np.ndarray, optional): SAM scores (parallel to masks).
            logits (np.ndarray, optional): SAM logits (unused for viz, just kept for completeness).
            point_coords, input_labels, input_box: optional SAM prompt info.
        """
        img = cv2.imread(img_path)
        if img is None:
            self._node.get_logger().error(f"Could not read image {img_path}")
            return

        out = img.copy()

        # --- SAM masks ---
        if masks is not None:
            for mask, score in zip(masks, scores):
                mask = mask.astype(np.uint8)
                overlay = np.zeros_like(out)
                overlay[mask > 0] = (30, 144, 255)
                out = cv2.addWeighted(out, 1.0, overlay, 0.5, 0)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
                cv2.putText(out, f"SAM {score:.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 144, 255), 2)

        # --- YOLO detections ---
        if detections is not None:
            for det in detections:
                if not det or "box" not in det: 
                    continue
                x1, y1, x2, y2 = map(int, det["box"])
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(out, f'{det.get("label","")}: {det.get("confidence",0):.2f}',
                            (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- SAM prompt overlays ---
        if input_box is not None:
            x0, y0, x1, y1 = map(int, input_box)
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2)
        if point_coords is not None and input_labels is not None:
            for (x, y), label in zip(point_coords, input_labels):
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.drawMarker(out, (int(x), int(y)), color,
                               markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)

        # Publish to ROS2
        msg = self._bridge.cv2_to_imgmsg(out, "bgr8")
        self._pub.publish(msg)
        rclpy.spin_once(self._node, timeout_sec=0.05)

    def close(self):
        self._node.destroy_node()
        if self._owns_context:
            rclpy.shutdown()
