#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory("stretch_audio_interface")
    default_template = "/home/ws/configs/audio_feedback.json"
    
    venv_arg = DeclareLaunchArgument(
        "venv_python",
        default_value="/home/ws/ros2_ws/src/stretch_audio_interface/venv/bin/python",
        description="Path to Python inside venv with Coqui TTS"
    )

    template_arg = DeclareLaunchArgument(
        "template_file",
        default_value=default_template,
        description="Path to JSON template file"
    )

    return LaunchDescription([
        venv_arg,
        template_arg,
        Node(
            package="stretch_audio_interface",
            executable="audio_feedback_node",   # <-- entry point, not .py
            name="audio_feedback_node",
            output="screen",
            arguments=[LaunchConfiguration("template_file")],
            prefix=[LaunchConfiguration("venv_python"), " "]
        )
    ])
