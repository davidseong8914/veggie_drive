#!/usr/bin/env python3

"""
Simple WildScenes inference script (no ROS2 dependencies)
Takes input point cloud file, runs inference, saves results
"""

import sys
import numpy as np
import torch
import os
from pathlib import Path

def load_model(model_path):
    """Load the WildScenes model"""
    print(f"Loading model from {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model
    model = torch.load(model_path, map_location='cpu')
    
    if isinstance(model, dict):
        if 'state_dict' in model:
            print("Model contains only state_dict - need complete model architecture")
            print("Using dummy inference for now...")
            return None  # We'll use dummy inference
        elif 'model' in model:
            return model['model']
    
    return model

def preprocess_points(points):
    """Preprocess point cloud for Cylinder3D"""
    # Filter points within range
    point_cloud_range = [-50, -50, -3, 50, 50, 1]
    
    mask = (
        (points[:, 0] >= point_cloud_range[0]) &
        (points[:, 0] <= point_cloud_range[3]) &
        (points[:, 1] >= point_cloud_range[1]) &
        (points[:, 1] <= point_cloud_range[4]) &
        (points[:, 2] >= point_cloud_range[2]) &
        (points[:, 2] <= point_cloud_range[5])
    )
    
    points_filtered = points[mask]
    
    if len(points_filtered) == 0:
        return np.array([]), np.array([])
    
    # Convert to cylindrical coordinates
    x, y, z = points_filtered[:, 0], points_filtered[:, 1], points_filtered[:, 2]
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta = (theta + 2 * np.pi) % (2 * np.pi)
    
    # Add intensity if available
    if points_filtered.shape[1] > 3:
        intensity = points_filtered[:, 3]
    else:
        intensity = np.ones(points_filtered.shape[0])
    
    # Create cylindrical features [rho, theta, z, intensity]
    features = np.stack([rho, theta, z, intensity], axis=1)
    
    return features, mask

def run_inference(points, model=None):
    """Run inference on point cloud"""
    print(f"Running inference on {len(points)} points")
    
    # Preprocess points
    processed_points, valid_mask = preprocess_points(points)
    
    if len(processed_points) == 0:
        print("No valid points after preprocessing")
        return None
    
    print(f"Processed {len(processed_points)} valid points")
    
    if model is None:
        # Dummy inference - random segmentation
        print("Using dummy inference (random segmentation)")
        predictions = np.random.randint(0, 19, len(processed_points))
    else:
        # Real inference would go here
        print("Real model inference not implemented yet")
        predictions = np.random.randint(0, 19, len(processed_points))
    
    # Map predictions back to original point cloud
    full_predictions = np.zeros(len(points), dtype=int)
    full_predictions[valid_mask] = predictions
    
    # Create segmented points [x, y, z, intensity, class]
    segmented_points = np.column_stack([
        points[:, 0],  # x
        points[:, 1],  # y
        points[:, 2],  # z
        points[:, 3] if points.shape[1] > 3 else np.zeros(len(points)),  # intensity
        full_predictions  # class
    ])
    
    return segmented_points

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 simple_inference.py <input_file.npy> <output_file.npy>")
        print("       python3 simple_inference.py <input_file.npy> <output_file.npy> [model_path]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model_path = sys.argv[3] if len(sys.argv) > 3 else '/workspace/wildscenes/pretrained_models/cylinder3d_wildscenes.pth'
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Model path: {model_path}")
    
    try:
        # Load input points
        print("Loading input points...")
        points = np.load(input_file)
        print(f"Loaded {len(points)} points with shape {points.shape}")
        
        # Load model
        model = load_model(model_path)
        
        # Run inference
        print("Running inference...")
        segmented_points = run_inference(points, model)
        
        if segmented_points is not None:
            # Save results
            print(f"Saving results to {output_file}")
            np.save(output_file, segmented_points)
            print(f"Saved {len(segmented_points)} segmented points")
            print("Inference complete!")
        else:
            print("Inference failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

