import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class ModeController(Node):
    """Switch the stretch_driver base mode.
    """

    def __init__(self):
        super().__init__('mode_controller_node')
        self.nav_client = self.create_client(Trigger, '/switch_to_navigation_mode')
        self.pos_client = self.create_client(Trigger, '/switch_to_position_mode')

    def _call(self, client, name: str) -> bool:
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f"Service {name} unavailable.")
            return False
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None or not result.success:
            msg = getattr(result, 'message', 'no response')
            self.get_logger().error(f"{name} failed: {msg}")
            return False
        return True

    def switch_to_navigation_mode(self) -> bool:
        return self._call(self.nav_client, '/switch_to_navigation_mode')

    def switch_to_position_mode(self) -> bool:
        return self._call(self.pos_client, '/switch_to_position_mode')


def main(args=None):
    rclpy.init(args=args)
    node = ModeController()
    ok = node.switch_to_position_mode()
    node.get_logger().info(f'Switch to position mode: {ok}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
