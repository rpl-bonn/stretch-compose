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
            'aligned_depth2color_subscriber = stretch_package.stretch_images.aligned_depth2color_subscriber:main',
            'compressed_image_subscriber = stretch_package.stretch_images.compressed_image_subscriber:main',
            'depth_image_subscriber = stretch_package.stretch_images.depth_image_subscriber:main',
            'rgb_image_subscriber = stretch_package.stretch_images.rgb_image_subscriber:main',
            'camera_info_subscriber = stretch_package.stretch_images.camera_info_subscriber:main',
            'navigation_image_subscriber = stretch_package.stretch_images.navigation_image_subscriber:main',
            'pointcloud_subscriber = stretch_package.stretch_pointclouds.pointcloud_subscriber:main',
            'odom_subscriber = stretch_package.stretch_state.odom_subscriber:main',
            'jointstate_subscriber = stretch_package.stretch_state.jointstate_subscriber:main',
            'image_service = stretch_package.stretch_images.image_service:main',
            'image_client = stretch_package.stretch_images.image_client:main',
            'pointcloud_service = stretch_package.stretch_pointclouds.pointcloud_service:main',
            'pointcloud_client = stretch_package.stretch_pointclouds.pointcloud_client:main',
            'localize_robot = stretch_package.localize_robot:main',
            'move_body = stretch_package.stretch_movement.move_body:main',
            'move_head = stretch_package.stretch_movement.move_head:main',
            'move_to_position = stretch_package.stretch_movement.move_to_position:main',
            'move_to_pose = stretch_package.stretch_movement.move_to_pose:main',
            'frame_transformer = stretch_package.stretch_state.frame_transformer:main',
        ],
    },
)
