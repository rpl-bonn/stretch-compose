import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class DetectionVisualizer:
  def __init__(self, node_name='detection_visualizer'):
    # Check if rclpy is already initialized
    if not rclpy.ok():
      rclpy.init()
      self._shutdown_rclpy = True
    else:
      self._shutdown_rclpy = False

    self.node = rclpy.create_node(node_name)
    self.image_pub = self.node.create_publisher(Image, 'visualized_detections', 10)
    self.bridge = CvBridge()

  def publish_visualized_detections(self, yolo_detections, original_image, obj=None, furniture=None, is_detected=None):
    """
    yolo_detections: list of dicts, each dict contains 'bbox' (x, y, w, h), 'score', 'class', 'mask' (optional)
    original_image: numpy array (OpenCV image)
    obj: optional string, name of the detected object
    furniture: optional string, name of the furniture
    is_detected: optional boolean, if set, overrides detection status for text display
    """
    vis_img = original_image.copy()

    detected = bool(yolo_detections)
    if is_detected is not None:
      detected = is_detected

    if not detected:
      # No detections, add "not detected" text
      if obj and furniture:
        text = f"{obj} not detected on {furniture}"
      elif obj:
        text = f"{obj} not detected"
      elif furniture:
        text = f"Not detected on {furniture}"
      else:
        text = "Not detected"
      font = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = 1.0
      thickness = 2
      color = (0, 0, 255)
      text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
      text_x = (vis_img.shape[1] - text_size[0]) // 2
      text_y = (vis_img.shape[0] + text_size[1]) // 2
      cv2.putText(vis_img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    else:
      # Detections present, add "detected" text
      if obj and furniture:
        text = f"{obj} detected on {furniture}"
      elif obj:
        text = f"{obj} detected"
      elif furniture:
        text = f"Detected on {furniture}"
      else:
        text = "Object detected"
      font = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = 1.0
      thickness = 2
      color = (0, 255, 0)
      text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
      text_x = (vis_img.shape[1] - text_size[0]) // 2
      text_y = text_size[1] + 10
      cv2.putText(vis_img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

      for det in yolo_detections:
        x, y, w, h = det['bbox']
        class_id = det.get('class', 0)
        score = det.get('score', 0)
        color = (0, 255, 0)
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
        label = f"{class_id}:{score:.2f}"
        cv2.putText(vis_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Draw mask if present
        if 'mask' in det and det['mask'] is not None:
          mask = det['mask']
          if mask.shape[:2] == vis_img.shape[:2]:
            colored_mask = np.zeros_like(vis_img)
            colored_mask[:, :, 1] = (mask * 255).astype(np.uint8)
            vis_img = cv2.addWeighted(vis_img, 1.0, colored_mask, 0.5, 0)

    ros_img = self.bridge.cv2_to_imgmsg(vis_img, encoding='bgr8')
    self.image_pub.publish(ros_img)

  def destroy(self):
    self.node.destroy_node()
    if self._shutdown_rclpy:
      rclpy.shutdown()