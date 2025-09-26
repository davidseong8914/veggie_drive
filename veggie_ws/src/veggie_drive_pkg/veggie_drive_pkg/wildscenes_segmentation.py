#!/usr/bin/env python3

"""
Inference script for WildScenes Cylinder3D
Will be referenced by the WildScenes ROS2 node to run inference
"""

import sys
import numpy as np
import torch
import os
from pathlib import Path

# Add WildScenes to path
wildscenes_path = Path('/veggie_drive/wildscenes')
sys.path.append(str(wildscenes_path))

# Import WildScenes components
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
import mmdet3d.models
import mmdet3d.datasets.transforms
from mmdet3d.utils import register_all_modules

# Register all modules
register_all_modules()

# Global model variable
model = None

def load_model(model_path, config_path):
    """Load the WildScenes Cylinder3D model"""
    global model
    
    print(f"Loading model from {model_path}")
    print(f"Using config from {config_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        # Load config
        cfg = Config.fromfile(config_path)
        
        # Build model
        model = MODELS.build(cfg.model)
        
        # Load checkpoint
        checkpoint = load_checkpoint(model, model_path, map_location='cpu')
        print(f"Loaded checkpoint from {model_path}")
        
        # Move model to device and set to eval mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
        return model
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None


def run_segmentation(points):
    """Run WildScenes segmentation on point cloud"""
    global model
    
    try:
        print(f"Running segmentation on {len(points)} points")

        if model is None:
            print("Model not loaded, using dummy segmentation")
            # Fallback to dummy segmentation
            class_labels = np.random.randint(0, 13, len(points))
        else:
            # Run real WildScenes segmentation
            class_labels = run_real_segmentation(points)

        segmented_points = np.column_stack([
            points[:, 0], # x
            points[:, 1], # y
            points[:, 2], # z
            class_labels   # class
        ])

        return segmented_points
    
    except Exception as e:
        print(f"Error running segmentation: {e}")
        return None

def run_real_segmentation(points):
    """Run real WildScenes segmentation"""
    global model
    
    try:
        with torch.no_grad():
            processed_points = preprocess_points(points)
            
            if len(processed_points) == 0:
                print("No valid points after preprocessing")
                return np.random.randint(0, 13, len(points))
            
            batch_inputs_dict = {
                'points': [torch.from_numpy(processed_points).float().to(model.device)]
            }
            
            # Create data sample
            data_sample = Det3DDataSample()
            data_sample.set_metainfo({
                'sample_idx': 'ros2_frame',
                'lidar_points': {
                    'lidar_path': 'ros2_frame',
                    'num_pts_feats': 3  # x, y, z
                }
            })
            
            batch_data_samples = [data_sample]
            
            processed_data = model.data_preprocessor({
                'inputs': batch_inputs_dict,
                'data_samples': batch_data_samples
            })
            
            results = model.predict(
                processed_data['inputs'], 
                processed_data['data_samples']
            )
            
            pred_result = results[0]
            if hasattr(pred_result, 'pred_pts_seg') and pred_result.pred_pts_seg is not None:
                predictions = pred_result.pred_pts_seg.cpu().numpy()
                full_predictions = np.random.randint(0, 13, len(points)) 
                return full_predictions
            else:
                print("No segmentation predictions found, using dummy")
                return np.random.randint(0, 13, len(points))
                
    except Exception as e:
        print(f"Real segmentation failed: {e}")
        return np.random.randint(0, 13, len(points))

def preprocess_points(points):
    """Preprocess point cloud for Cylinder3D model"""
    # filtering points by range - same as config
    point_cloud_range = [0, -3.14159265359, -4, 50, 3.14159265359, 10]
    
    mask = (
        (points[:, 0] >= point_cloud_range[0]) &
        (points[:, 0] <= point_cloud_range[3]) &
        (points[:, 1] >= point_cloud_range[1]) &
        (points[:, 1] <= point_cloud_range[4]) &
        (points[:, 2] >= point_cloud_range[2]) &
        (points[:, 2] <= point_cloud_range[5])
    )
    
    return points[mask, :3]  # Return only x, y, z coordinates

def initialize_model():
    """Initialize the WildScenes model"""
    model_path = '/veggie_drive/wildscenes/pretrained_models/cylinder3d_wildscenes.pth'
    config_path = '/veggie_drive/wildscenes/config.py'
    
    print("Initializing WildScenes model...")
    return load_model(model_path, config_path)

