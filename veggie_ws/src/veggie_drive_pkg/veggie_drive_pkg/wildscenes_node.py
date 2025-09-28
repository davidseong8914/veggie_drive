#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np
import time
from . import wildscenes_segmentation

class WildscenesSegmentation(Node):
    def __init__(self):
        super().__init__('wildscenes_segmentation')
        self.get_logger().info('Wildscenes node started')
        
        # Rate limiting
        self.last_process_time = 0
        self.min_interval = 0.5  # Process at most every 0.5 seconds
        
        wildscenes_segmentation.initialize_model()
        self.get_logger().info('Wildscenes node initialized')

        self.subscription = self.create_subscription(
            PointCloud2, 
            '/livox/lidar',
            self.lidar_callback,
            1)  # Reduced queue size to avoid message drops
        
        self.publisher = self.create_publisher(
            PointCloud2, 
            'wildscenes_segmented', 
            1)  # Reduced queue size to avoid message drops

    def lidar_callback(self, msg: PointCloud2):
        # Rate limiting to avoid overwhelming the system
        current_time = time.time()
        if current_time - self.last_process_time < self.min_interval:
            return
        
        self.last_process_time = current_time
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
            # Debug: Check segmented points structure
            self.get_logger().info(f"Segmented points shape: {segmented_points.shape}")
            self.get_logger().info(f"Sample RGB values: {segmented_points[0, 3:6]}")
            
            segmented_points_msg = self.convert_to_pointcloud2(segmented_points, msg.header)
            self.publisher.publish(segmented_points_msg)
            self.get_logger().info("Published segmented point cloud")

        else:
            self.get_logger().warning("Segmentation failed")

    def convert_to_pointcloud2(self, segmented_points, header):
        fields = [
            point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='rgb', offset=12, datatype=point_cloud2.PointField.UINT32, count=1),
            point_cloud2.PointField(name='class', offset=16, datatype=point_cloud2.PointField.UINT8, count=1),
        ]

        points_list = []
        for i, point in enumerate(segmented_points):
            if i < 3:  # Debug first 3 points
                self.get_logger().info(f"Point {i}: x={point[0]:.2f}, y={point[1]:.2f}, z={point[2]:.2f}, r={point[3]}, g={point[4]}, b={point[5]}, class={point[6]}")
            
            # Pack RGB into single UINT32: (r << 16) | (g << 8) | b
            r, g, b = int(point[3]), int(point[4]), int(point[5])
            rgb_packed = (r << 16) | (g << 8) | b
            
            points_list.append([
                float(point[0]), # x
                float(point[1]), # y
                float(point[2]), # z
                rgb_packed,      # rgb (packed)
                int(point[6])    # class
            ])

        return point_cloud2.create_cloud(header, fields, points_list)


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