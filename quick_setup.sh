#!/bin/bash

# ROS2 setup for Docker container
source /opt/ros/humble/setup.bash
cd /home/patrick/Desktop/veggie_drive/veggie_ws
source install/setup.bash
cd /home/patrick/Desktop/veggie_drive

echo "ROS2 environment ready!"
echo "ROS_DISTRO: $ROS_DISTRO"
echo ""
echo "For WildScenes ROS2 Inference:"
echo "  ros2 launch veggie_drive_pkg wildscenes_launch.py"
echo "  ros2 bag play data/livox_data_jetson --rate 0.5 --loop"
echo "  rviz2 -d segmented_points.rviz"
