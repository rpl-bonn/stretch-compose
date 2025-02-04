from setuptools import find_packages, setup

package_name = 'stretch_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='schmied1',
    maintainer_email='s6yaschm@uni-bonn.de',
    description='Package for communication with the Stretch 3 robot from HelloRobot.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'compressed_gripper_image_subscriber = stretch_package.compressed_gripper_image_subscriber:main',
            'compressed_image_subscriber = stretch_package.compressed_image_subscriber:main',
            'image_subscriber = stretch_package.image_subscriber:main',
            'pointcloud_subscriber = stretch_package.pointcloud_subscriber:main',
            'odom_subscriber = stretch_package.odom_subscriber:main',
            'jointstate_subscriber = stretch_package.jointstate_subscriber:main',
            'image_service = stretch_package.image_service:main',
            'image_client = stretch_package.image_client:main',
            'pointcloud_service = stretch_package.pointcloud_service:main',
            'pointcloud_client = stretch_package.pointcloud_client:main',
            'localize_robot = stretch_package.localize_robot:main',
            'stretch_client = stretch_package.stretch_client:main',
            'move_head = stretch_package.move_head:main',
            'move_to_position = stretch_package.move_to_position:main',
            'move_to_pose = stretch_package.move_to_pose:main',
        ],
    },
)
