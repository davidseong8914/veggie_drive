#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    enable_realsense = LaunchConfiguration('enable_realsense')
    enable_livox = LaunchConfiguration('enable_livox')

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ),
        launch_arguments={
            'pointcloud.enable': 'true',
            'rgb_camera.enable': 'true',
            'depth_module.profile': '640x480x30'
        }.items(),
        condition=IfCondition(enable_realsense)
    )

    livox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('livox_ros_driver2'),
                'launch_ROS2',
                'msg_MID360_launch.py'
            )
        ),
        condition=IfCondition(enable_livox)
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'enable_realsense',
            default_value='false',
            description='Launch Intel RealSense driver'
        ),
        DeclareLaunchArgument(
            'enable_livox',
            default_value='false',
            description='Launch Livox driver'
        ),
        realsense_launch,
        livox_launch,
        # Static transform publisher to create livox_frame
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='livox_frame_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'livox_frame'],
            output='screen'
        ),
        # WildScenes segmentation node
        Node(
            package='veggie_drive_pkg',
            executable='wildscenes_node',
            name='wildscenes_segmentation',
            output='screen',
            parameters=[{
                'use_sim_time': False
            }]
        )
    ])