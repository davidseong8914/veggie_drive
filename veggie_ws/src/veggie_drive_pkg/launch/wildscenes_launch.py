#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
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
