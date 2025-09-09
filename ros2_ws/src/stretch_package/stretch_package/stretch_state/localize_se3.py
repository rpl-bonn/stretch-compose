import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

class FunMapLocalizer(Node):
    def __init__(self):
        super().__init__('funmap_localizer')
        self.cli_global = self.create_client(Trigger, '/funmap/localize_robot')
        self.cli_local = self.create_client(Trigger, '/funmap/localize_near_map')

        while not self.cli_global.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /funmap/localize_robot...')
        while not self.cli_local.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /funmap/localize_near_map...')

    def global_localize(self):
        req = Trigger.Request()
        future = self.cli_global.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success, future.result().message

    def local_localize(self):
        req = Trigger.Request()
        future = self.cli_local.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success, future.result().message


def main(args=None):
    rclpy.init(args=args)
    node = FunMapLocalizer()
    success, msg = node.global_localize()
    node.get_logger().info(f'Global localization: {success}, {msg}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
