#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np
from . import wildscenes_segmentation

class WildscenesSegmentation(Node):
    def __init__(self):
        super().__init__('wildscenes_segmentation')
        self.get_logger().info('Wildscenes node started')
        
        wildscenes_segmentation.initialize_model()
        self.get_logger().info('Wildscenes node initialized')

        self.subscription = self.create_subscription(
            PointCloud2, 
            '/livox/lidar',
            self.lidar_callback,
            10)
        
        self.publisher = self.create_publisher(
            PointCloud2, 
            'wildscenes_segmented', 
            10)

    def lidar_callback(self, msg: PointCloud2):
        self.get_logger().info("Received PointCloud2 message")

        points_list = list(point_cloud2.read_points(
            # msg, field_names=("x", "y", "z", "intensity"), skip_nans=True
            msg, field_names=("x", "y", "z"), skip_nans=True
        ))

        if len(points_list) == 0:
            self.get_logger().warning("No points received")
            return

        # convert points to numpy array
        # points = np.array([(p[0], p[1], p[2], p[3]) for p in points_list], dtype=np.float32)
        points = np.array([(p[0], p[1], p[2]) for p in points_list], dtype=np.float32)

        self.get_logger().info(f"Processing {len(points)} points")

        # run segmentation using wildscenes_segmentation.py - run_segmentation function
        segmented_points = wildscenes_segmentation.run_segmentation(points)

        if segmented_points is not None:
            self.get_logger().info(f"Segmented {len(segmented_points)} points")

            # convert segmented points to ROS2 PointCloud2
            segmented_points_msg = self.convert_to_pointcloud2(segmented_points, msg.header)
            self.publisher.publish(segmented_points_msg)

        else:
            self.get_logger().warning("Segmentation failed")

    def convert_to_pointcloud2(self, segmented_points, header):
        # convert segmented points to ROS2 PointCloud2
        fields = [
            point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1),
            # point_cloud2.PointField(name='intensity', offset=12, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='class', offset=16, datatype=point_cloud2.PointField.FLOAT32, count=1),
        ]

        points_list = []
        for point in segmented_points:
            points_list.append([
                float(point[0]), # x
                float(point[1]), # y
                float(point[2]), # z
                # float(point[3]), # intensity
                float(point[3]) # class
            ])

        cloud_msg = point_cloud2.create_cloud(header, fields, points_list)
        return cloud_msg


def main(args=None):
    rclpy.init(args=args)

    try:
        node = WildscenesSegmentation()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()