from scripts.my_robot_scripts.test_scripts.hold_node import hold_door
from scripts.my_robot_scripts.test_scripts.home_pose import execute_home_pose
from scripts.my_robot_scripts.test_scripts.release_node import release_door
import rclpy
import time

def main(args=None):
    # execute_home_pose()
    # print("Executed Home Pose")
    # time.sleep(2)

    hold_door()
    print("Executed Hold Door")
    time.sleep(5)

    release_door()
    print("Executed Release Door")
    # rclpy.shutdown()
    
if __name__ == '__main__':
    main()