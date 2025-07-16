import rclpy
from rclpy.node import Node
import tf2_ros
from rclpy.time import Time
import numpy as np
from scipy.spatial.transform import Rotation

class FrameTransformer(Node):
    
    def __init__(self, buffer: tf2_ros.Buffer, ):
        super().__init__('frame_transformer_node')
        #self.get_logger().info("Transformer initialized.")
        self.buffer = buffer
        self.done = False
            
    def get_tf_matrix(self, target_frame: str, source_frame: str): 
        while rclpy.ok():
            try:
                if not self.buffer.can_transform(target_frame, source_frame, Time()):
                    self.get_logger().warn(f'Transform from {source_frame} to {target_frame} is not available yet. Retrying...')
                    rclpy.spin_once(self, timeout_sec=1.0)
                    continue
                tf = self.buffer.lookup_transform(target_frame, source_frame, Time())
                t = np.eye(4)
                q = tf.transform.rotation
                x = tf.transform.translation
                t[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                t[:3, 3] = [x.x, x.y, x.z]
                self.done = True
                self.get_logger().info(f"Transform from {source_frame} to {target_frame} successful.")
                return t #point @ t.T #t.dot(point)[:3]
            except Exception as e:
                self.get_logger().error(f"Transform from {source_frame} to {target_frame} failed: {e}")          