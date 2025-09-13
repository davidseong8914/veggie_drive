#!/usr/bin/env python3

""" Cylinder 3D inference node for ROS2 - using Livox LiDAR (currently) can also switch to Hesai but we will see """

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point32
from sensor_msgs_py import point_cloud2
import numpy as np
import torch
import yaml
import time
import os
from pathlib import Path

# include cylinder 3d model
# import the utils we copied from wildscenes
from utils3d import cidx_2_rgb


class Cylinder3DInferenceNode(Node):
    def __init__(self):
        super().__init__('cylinder3d_inference')

        # parameters
        self.declare_parameter('model_path', 'wildscenes/pretrained_models/cylinder3d_wildscenes.pth')
        self.declare_parameter('config_path', 'wildscenes/config.py')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('voxel_size', [0.1, 0.1, 0.2])  # fixed typo
        self.declare_parameter('point_cloud_range', [-50, -50, -3, 50, 50, 1])  # may need adjustments

        # get parameters
        self.model_path = self.get_parameter('model_path').value
        self.config_path = self.get_parameter('config_path').value
        self.device = torch.device(self.get_parameter('device').value if torch.cuda.is_available() else 'cpu')
        self.voxel_size = self.get_parameter('voxel_size').value
        self.point_cloud_range = self.get_parameter('point_cloud_range').value

        # wildscenes class names (19 classes)
        self.class_names = [
            'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
            'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
            'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
            'pole', 'traffic-sign'
        ]

        # load model
        self.load_model()

        # subscribe to livox point cloud
        self.subscription = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.lidar_callback,
            10
        )

        # publish segmented point cloud
        self.publisher = self.create_publisher(
            PointCloud2,
            '/cylinder3d/segmented_points',
            10 
        )

        # inference time log
        self.inference_times = []
        self.frame_count = 0
        self.get_logger().info('Cylinder 3D inference node initialized')
        self.get_logger().info(f'Using device: {self.device}')

    # load Cylinder3D WildScenes model
    def load_model(self):
        try: 
            if not os.path.exists(self.model_path):
                self.get_logger().error(f'Model file not found: {self.model_path}')
                raise FileNotFoundError(f'Model file not found: {self.model_path}')
            
            self.get_logger().info(f'Loading model from {self.model_path}')
            
            # load the complete pretrained model
            self.model = torch.load(self.model_path, map_location=self.device)
            
            # if loaded object is a checkpoint dict, extract the model
            if isinstance(self.model, dict):
                if 'model' in self.model:
                    self.model = self.model['model']
                elif 'state_dict' in self.model:
                    self.get_logger().error("Only state_dict found - need complete model architecture")
                    raise ValueError("Please save the complete model, not just state_dict")
            
            # ensure model is in eval mode and on correct device
            self.model.to(self.device)
            self.model.eval()
            
            self.get_logger().info('Model loaded successfully')
            
            # test the model with dummy input
            self.test_model_inference()

        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            self.model = None
            self.get_logger().warn('Model loading failed - will use dummy predictions for testing')

    # test model with dummy input to verify it works
    def test_model_inference(self):
        try:
            # create dummy input matching expected format
            dummy_points = torch.randn(100, 4).to(self.device)  # [N, 4] for [x,y,z,intensity]
            
            with torch.no_grad():
                # test inference - adjust based on your model's expected input format
                output = self.model(dummy_points.unsqueeze(0))  # add batch dimension if needed
                self.get_logger().info(f'Model test successful - output type: {type(output)}')
                
        except Exception as e:
            self.get_logger().warn(f'Model test failed: {e} - but will continue')

    # pre processing point cloud before inference
    def pre_process_point_cloud(self, points):
        # filter points within range
        mask = (
            (points[:, 0] >= self.point_cloud_range[0]) &
            (points[:, 0] <= self.point_cloud_range[3]) &
            (points[:, 1] >= self.point_cloud_range[1]) &
            (points[:, 1] <= self.point_cloud_range[4]) &
            (points[:, 2] >= self.point_cloud_range[2]) &
            (points[:, 2] <= self.point_cloud_range[5])
        )

        points_filtered = points[mask]
        
        if len(points_filtered) == 0:
            return torch.empty((0, 4)).to(self.device), np.array([])
        
        # convert to cylindrical coordinates for Cylinder3D
        x, y, z = points_filtered[:, 0], points_filtered[:, 1], points_filtered[:, 2]
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # normalize theta to [0, 2π]
        theta = (theta + 2 * np.pi) % (2 * np.pi)

        # adding intensity if available
        if points_filtered.shape[1] > 3:
            intensity = points_filtered[:, 3]
            # create cylindrical features [rho, theta, z, intensity]
            features = np.stack([rho, theta, z, intensity], axis=1)
        # use dummy intensity if unavailable
        else:
            self.get_logger().warning('No intensity data available, using dummy intensity')
            intensity = np.ones(points_filtered.shape[0])
            # create cylindrical features [rho, theta, z, intensity]
            features = np.stack([rho, theta, z, intensity], axis=1)

        return torch.from_numpy(features).float().to(self.device), mask

    # process lidar point cloud
    def lidar_callback(self, msg):
        start_time = time.time()
        self.frame_count += 1

        try:
            # convert ROS2 PointCloud2 to numpy array
            points = np.array(list(point_cloud2.read_points(
                # outputs of Livox LiDAR
                msg, field_names=("x", "y", "z", "intensity"), skip_nans=True
            )))

            if len(points) == 0:
                self.get_logger().warning('No points received from LiDAR')
                return

            # preprocess points
            processed_points, valid_mask = self.pre_process_point_cloud(points)
            
            if processed_points.shape[0] == 0:
                self.get_logger().warning('No valid points after filtering')
                return

            # run inference
            with torch.no_grad():
                if self.model is not None:
                    try:
                        # prepare input for pretrained model
                        model_input = processed_points.unsqueeze(0)  # add batch dimension
                        
                        # run inference
                        output = self.model(model_input)
                        
                        # process output based on model's output format
                        if isinstance(output, torch.Tensor):
                            if output.dim() > 1:
                                predictions = torch.argmax(output, dim=-1).squeeze().cpu().numpy()
                            else:
                                predictions = output.squeeze().cpu().numpy()
                        elif isinstance(output, (list, tuple)):
                            predictions = output[0]
                            if isinstance(predictions, torch.Tensor):
                                if predictions.dim() > 1:
                                    predictions = torch.argmax(predictions, dim=-1).cpu().numpy()
                                else:
                                    predictions = predictions.cpu().numpy()
                        else:
                            raise ValueError(f"Unexpected output type: {type(output)}")
                        
                        # ensure predictions are the right shape
                        if predictions.shape[0] != processed_points.shape[0]:
                            self.get_logger().warn(f"Prediction shape mismatch: {predictions.shape[0]} vs {processed_points.shape[0]}")
                            predictions = np.random.randint(0, len(self.class_names), processed_points.shape[0])
                            
                    except Exception as e:
                        self.get_logger().error(f'Model inference failed: {e}')
                        # fallback to dummy predictions
                        predictions = np.random.randint(0, len(self.class_names), processed_points.shape[0])
                else:
                    # dummy predictions for testing when model fails to load
                    predictions = np.random.randint(0, len(self.class_names), processed_points.shape[0])

            # map predictions back to original point cloud
            full_predictions = np.zeros(len(points), dtype=int)
            full_predictions[valid_mask] = predictions

            # publish segmented cloud
            self.publish_segmented_cloud(points, full_predictions, msg.header)

            # log inference performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            # every 30 frames, print performance metrics
            if self.frame_count % 30 == 0:
                if len(self.inference_times) > 0:
                    avg_time = np.mean(self.inference_times[-30:])
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    self.get_logger().info(
                        f'Frame {self.frame_count}: Avg inference time: {avg_time:.3f}s, FPS: {fps:.1f}, Points: {len(points)}'
                    )

        except Exception as e:
            self.get_logger().error(f'Callback failed: {e}')

    # publish segmented point cloud
    def publish_segmented_cloud(self, original_points, predictions, header):
        try:
            # get class colors from wildscenes utils
            class_colors = self.get_wildscenes_colors()
            
            # create segmented points with colors
            segmented_points = []
            for i, (point, pred) in enumerate(zip(original_points, predictions)):
                if len(point) >= 3:  # ensure we have x, y, z
                    x, y, z = float(point[0]), float(point[1]), float(point[2])
                    intensity = float(point[3]) if len(point) > 3 else 0.0
                    
                    # get color for this class
                    color = class_colors[pred % len(class_colors)]
                    
                    segmented_points.append([
                        x, y, z, intensity, float(pred),
                        int(color[0]), int(color[1]), int(color[2])
                    ])
            
            if not segmented_points:
                return
            
            # create PointCloud2 message with additional fields
            fields = [
                point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
                point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
                point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1),
                point_cloud2.PointField(name='intensity', offset=12, datatype=point_cloud2.PointField.FLOAT32, count=1),
                point_cloud2.PointField(name='class', offset=16, datatype=point_cloud2.PointField.FLOAT32, count=1),
                point_cloud2.PointField(name='r', offset=20, datatype=point_cloud2.PointField.UINT8, count=1),
                point_cloud2.PointField(name='g', offset=21, datatype=point_cloud2.PointField.UINT8, count=1),
                point_cloud2.PointField(name='b', offset=22, datatype=point_cloud2.PointField.UINT8, count=1),
            ]
            
            cloud_msg = point_cloud2.create_cloud(header, fields, segmented_points)
            self.publisher.publish(cloud_msg)
            
        except Exception as e:
            self.get_logger().error(f'Failed to publish segmented cloud: {e}')

    # get colors from wildscenes utils if available, otherwise use default colors
    def get_wildscenes_colors(self):
        try:
            # try to use wildscenes color mapping
            colors = []
            for i in range(len(self.class_names)):
                if i in cidx_2_rgb:
                    colors.append(cidx_2_rgb[i])
                else:
                    colors.append([128, 128, 128])  # default gray
            return colors
        except:
            # fallback color scheme
            return [
                [0, 0, 0],        # unlabeled - black
                [245, 150, 100],  # car - orange
                [245, 230, 100],  # bicycle - yellow
                [150, 60, 30],    # motorcycle - brown
                [180, 30, 80],    # truck - purple
                [255, 0, 0],      # other-vehicle - red
                [30, 30, 255],    # person - blue
                [200, 40, 255],   # bicyclist - magenta
                [90, 30, 150],    # motorcyclist - dark purple
                [255, 0, 255],    # road - magenta
                [255, 150, 255],  # parking - light magenta
                [75, 0, 75],      # sidewalk - dark magenta
                [75, 0, 175],     # other-ground - dark blue
                [0, 200, 255],    # building - cyan
                [50, 120, 255],   # fence - light blue
                [0, 175, 0],      # vegetation - green
                [0, 60, 135],     # trunk - dark blue
                [80, 240, 150],   # terrain - light green
                [150, 240, 255],  # pole - light cyan
                [0, 0, 255]       # traffic-sign - blue
            ]


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = Cylinder3DInferenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()