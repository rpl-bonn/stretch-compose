import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np

class GraspVisualizer(Node):
    def __init__(self):
        super().__init__('grasp_visualizer')
        self.pub = self.create_publisher(MarkerArray, '/grasps_markers', 10)

    def publish_grasps(self, grasp_list):
        ma = MarkerArray()
        for i, g in enumerate(grasp_list.grasps):
            # Make a frame at the grasp position
            pos = np.array([g.position.x, g.position.y, g.position.z])

            # Each vector is already a unit axis from GPD
            approach = np.array([g.approach.x, g.approach.y, g.approach.z])
            binormal = np.array([g.binormal.x, g.binormal.y, g.binormal.z])
            axis = np.array([g.axis.x, g.axis.y, g.axis.z])

            # Helper to make arrows
            def make_arrow(vec, color, ns):
                m = Marker()
                m.header = grasp_list.header
                m.type = Marker.ARROW
                m.action = Marker.ADD
                m.ns = ns
                m.id = i
                m.scale.x = 0.005  # shaft diameter
                m.scale.y = 0.01   # head diameter
                m.scale.z = 0.02   # head length
                m.color.r, m.color.g, m.color.b, m.color.a = color
                start = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
                end = Point(x=float(pos[0]+0.05*vec[0]),
                            y=float(pos[1]+0.05*vec[1]),
                            z=float(pos[2]+0.05*vec[2]))
                m.points = [start, end]
                return m

            ma.markers.append(make_arrow(approach, (1,0,0,1), 'approach'))  # red
            ma.markers.append(make_arrow(binormal, (0,1,0,1), 'binormal'))  # green
            ma.markers.append(make_arrow(axis, (0,0,1,1), 'axis'))          # blue

        self.pub.publish(ma)

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import tf_transformations

class GraspTFPublisher(Node):
    def __init__(self):
        super().__init__('grasp_tf_publisher')
        self.br = TransformBroadcaster(self)

    def publish_grasp_tf(self, g, frame_id, name="grasp"):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = frame_id
        t.child_frame_id = name

        t.transform.translation.x = g.position.x
        t.transform.translation.y = g.position.y
        t.transform.translation.z = g.position.z

        # GPD gives you 3 axes: approach, binormal, axis.
        # Build rotation matrix [axis binormal approach] or similar convention.
        R = np.vstack([
            [g.axis.x, g.binormal.x, g.approach.x],
            [g.axis.y, g.binormal.y, g.approach.y],
            [g.axis.z, g.binormal.z, g.approach.z]
        ])
        q = tf_transformations.quaternion_from_matrix(np.vstack([
            np.hstack([R, np.array([[0],[0],[0]])]),
            np.array([0,0,0,1])
        ]))
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q

        self.br.sendTransform(t)
