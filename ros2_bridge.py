#!/usr/bin/env python3

"""
ROS2 Bridge for WildScenes Cylinder3D Inference
- Subscribes to point cloud topics
- Calls WildScenes model for inference
- Publishes segmented results
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np
import subprocess
import tempfile
import os
import json
import time
from pathlib import Path

class WildScenesBridge(Node):
    def __init__(self):
        super().__init__('wildscenes_bridge')
        
        # Parameters
        self.declare_parameter('input_topic', '/livox/lidar')
        self.declare_parameter('output_topic', '/cylinder3d/segmented_points')
        self.declare_parameter('model_path', '/home/patrick/Desktop/WildScenes/pretrained_models/cylinder3d_wildscenes.pth')
        self.declare_parameter('use_docker', True)
        
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.model_path = self.get_parameter('model_path').value
        self.use_docker = self.get_parameter('use_docker').value
        
        # Create temp directory for data exchange
        self.temp_dir = Path(tempfile.mkdtemp(prefix='wildscenes_'))
        self.get_logger().info(f'Using temp directory: {self.temp_dir}')
        
        # Subscribe to input point cloud
        self.subscription = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.point_cloud_callback,
            10
        )
        
        # Publish segmented point cloud
        self.publisher = self.create_publisher(
            PointCloud2,
            self.output_topic,
            10
        )
        
        # Statistics
        self.frame_count = 0
        self.inference_times = []
        
        self.get_logger().info(f'WildScenes Bridge initialized')
        self.get_logger().info(f'Input topic: {self.input_topic}')
        self.get_logger().info(f'Output topic: {self.output_topic}')
        self.get_logger().info(f'Model path: {self.model_path}')
        self.get_logger().info(f'Use Docker: {self.use_docker}')

    def point_cloud_callback(self, msg):
        start_time = time.time()
        self.frame_count += 1
        
        try:
            # Convert ROS2 PointCloud2 to numpy array
            points_list = list(point_cloud2.read_points(
                msg, field_names=("x", "y", "z", "intensity"), skip_nans=True
            ))
            
            if len(points_list) == 0:
                self.get_logger().warning('No points received')
                return
                
            # Convert structured array to regular float32 array
            points = np.array([(p[0], p[1], p[2], p[3]) for p in points_list], dtype=np.float32)
            self.get_logger().info(f'Processing {len(points)} points')
            
            # Run inference
            segmented_points = self.run_inference(points)
            
            if segmented_points is not None:
                # Publish segmented results
                self.publish_segmented_cloud(segmented_points, msg.header)
                
                # Log performance
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                
                if self.frame_count % 30 == 0:
                    avg_time = np.mean(self.inference_times[-30:])
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    self.get_logger().info(
                        f'Frame {self.frame_count}: Avg time: {avg_time:.3f}s, FPS: {fps:.1f}'
                    )
            else:
                self.get_logger().warning('Inference failed')
                
        except Exception as e:
            self.get_logger().error(f'Callback failed: {e}')

    def run_inference(self, points):
        """Run WildScenes inference on point cloud"""
        try:
            if self.use_docker:
                return self.run_inference_docker(points)
            else:
                return self.run_inference_direct(points)
        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')
            return None

    def run_inference_docker(self, points):
        """Run inference using Docker container"""
        # Save points to temporary file
        input_file = self.temp_dir / f'input_{self.frame_count}.npy'
        output_file = self.temp_dir / f'output_{self.frame_count}.npy'
        
        np.save(input_file, points)
        
        # Run Docker inference
        cmd = [
            'docker', 'run', '--rm',
            '--gpus', 'all',
            '--network', 'host',
            '-v', f'{self.temp_dir}:/workspace/data',
            '-v', f'{self.model_path}:/workspace/model.pth',
            'wildscenes:latest',
            'python3', '-c', f'''
import numpy as np
import torch
import sys
sys.path.append("/workspace/veggie_drive")

# Load points
points = np.load("/workspace/data/{input_file.name}")

# Simple inference (placeholder - you'd implement the actual model loading here)
# For now, just return dummy segmentation
segmented = np.column_stack([points, np.random.randint(0, 19, len(points))])

# Save results
np.save("/workspace/data/{output_file.name}", segmented)
print("Inference complete")
'''
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and output_file.exists():
            segmented_points = np.load(output_file)
            # Clean up files
            input_file.unlink(missing_ok=True)
            output_file.unlink(missing_ok=True)
            return segmented_points
        else:
            self.get_logger().error(f'Docker inference failed: {result.stderr}')
            return None

    def run_inference_direct(self, points):
        """Run inference directly (if ROS2 and WildScenes are compatible)"""
        # This would be the direct approach if we had ROS2 working in the same environment
        # For now, return dummy results
        segmented = np.column_stack([points, np.random.randint(0, 19, len(points))])
        return segmented

    def publish_segmented_cloud(self, segmented_points, header):
        """Publish segmented point cloud to ROS2"""
        try:
            # Create PointCloud2 message
            fields = [
                point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
                point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
                point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1),
                point_cloud2.PointField(name='intensity', offset=12, datatype=point_cloud2.PointField.FLOAT32, count=1),
                point_cloud2.PointField(name='class', offset=16, datatype=point_cloud2.PointField.FLOAT32, count=1),
            ]
            
            # Convert segmented points to list format
            points_list = []
            for point in segmented_points:
                points_list.append([
                    float(point[0]),  # x
                    float(point[1]),  # y
                    float(point[2]),  # z
                    float(point[3]),  # intensity
                    float(point[4])   # class
                ])
            
            cloud_msg = point_cloud2.create_cloud(header, fields, points_list)
            self.publisher.publish(cloud_msg)
            
        except Exception as e:
            self.get_logger().error(f'Failed to publish segmented cloud: {e}')

    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = WildScenesBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if 'node' in locals():
            node.cleanup()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
