#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import threading

class AudioFeedbackInterface:
    def __init__(self, node: Node = None, node_name: str = "audio_feedback_interface"):
        self._own_node = False

        if node is None:
            if not rclpy.ok():
                rclpy.init()
            self.node = Node(node_name)
            self._own_node = True
            # background spin so publishers work
            self._spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
            self._spin_thread.start()
        else:
            self.node = node

        self.pub = self.node.create_publisher(String, "audio_feedback", 10)

    def send(self, message_type: str, **kwargs):
        payload = {"message_type": message_type}
        payload.update(kwargs)
        msg = String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)
        self.node.get_logger().info(f"Published: {msg.data}")
        
    def destroy(self):
        if self._own_node:
            self.node.destroy_node()
            rclpy.shutdown()

def main(args=None):
    iface = AudioFeedbackInterface()
    iface.send("greeting")
    import time; time.sleep(2)  # allow time for DDS
    iface.destroy()
    
if __name__ == '__main__':
    main()