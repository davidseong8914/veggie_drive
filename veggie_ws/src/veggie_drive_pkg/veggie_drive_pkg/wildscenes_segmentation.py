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

def get_class_colors():
    """Get consistent colors for each terrain class (matching WildScenes config)"""
    return {
        0: [0.0, 0.8, 0.0],      # bush - bright green
        1: [0.6, 0.4, 0.2],      # dirt - brown
        2: [0.8, 0.8, 0.0],      # fence - yellow
        3: [0.0, 0.6, 0.0],      # grass - dark green
        4: [0.7, 0.7, 0.7],      # gravel - light gray
        5: [0.4, 0.2, 0.0],      # log - dark brown
        6: [0.3, 0.2, 0.1],      # mud - dark brown
        7: [1.0, 0.0, 1.0],      # object - magenta
        8: [0.5, 0.5, 0.5],      # other-terrain - gray
        9: [0.4, 0.4, 0.4],      # rock - dark gray
        10: [0.8, 0.4, 0.0],     # structure - orange
        11: [0.0, 0.4, 0.0],     # tree-foliage - forest green
        12: [0.2, 0.1, 0.0],     # tree-trunk - very dark brown
        255: [1.0, 1.0, 1.0],    # unsegmented - white
    }

def get_class_names():
    """Get class names matching WildScenes config"""
    return [
        "bush", "dirt", "fence", "grass", "gravel", "log", "mud",
        "object", "other-terrain", "rock", "structure", "tree-foliage", "tree-trunk"
    ]

def get_class_name(class_id):
    """Get class name for a given class ID"""
    class_names = get_class_names()
    if class_id == 255:
        return "unsegmented"
    elif class_id < len(class_names):
        return class_names[class_id]
    else:
        return "unknown"

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
    
    print(f"Loading model from: {model_path}", flush=True)
    print(f"Loading config from: {config_path}", flush=True)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}", flush=True)
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}", flush=True)
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print("Model and config files found, proceeding with loading...", flush=True)

    try:
        # Clear CUDA cache before loading
        if torch.cuda.is_available():
            print("CUDA available, clearing cache...", flush=True)
            torch.cuda.empty_cache()
        else:
            print("CUDA not available, using CPU...", flush=True)
        
        # Load config
        print("Loading config...", flush=True)
        cfg = Config.fromfile(config_path)
        print("Config loaded successfully", flush=True)
        
        # Build model
        print("Building model...", flush=True)
        model = MODELS.build(cfg.model)
        print("Model built successfully", flush=True)
        
        # Load checkpoint
        print("Loading checkpoint...", flush=True)
        checkpoint = load_checkpoint(model, model_path, map_location='cpu')
        print("Checkpoint loaded successfully", flush=True)
        
        # Move model to device and set to eval mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}", flush=True)
        
        try:
            print("Moving model to device...", flush=True)
            model = model.to(device)
            model.eval()
            print("Model moved to device and set to eval mode", flush=True)
            
            # Clear CUDA cache after loading
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                print("CUDA cache cleared after loading", flush=True)
            
            print("Model loading completed successfully!", flush=True)
            return model
            
        except RuntimeError as cuda_error:
            print(f"CUDA error: {cuda_error}", flush=True)
            if "CUDA" in str(cuda_error):
                print("Falling back to CPU...", flush=True)
                device = torch.device("cpu")
                model = model.to(device)
                model.eval()
                print("Model moved to CPU successfully", flush=True)
                return model
            else:
                raise cuda_error
        
    except Exception as e:
        print(f"Error loading model: {e}", flush=True)
        return None


def run_segmentation(points):
    """Run WildScenes segmentation on point cloud"""
    global model
    
    try:
        if model is None:
            print("Model not loaded - using white for all points", flush=True)
            n_points = len(points)
            class_labels = np.full(n_points, 255, dtype=int)  # 255 = white/unsegmented
        else:
            print("Running Cylinder3D segmentation...", flush=True)
            try:
                class_labels = run_cylinder3d_segmentation(points, model)
                if class_labels is None:
                    print("Segmentation failed - using white for all points", flush=True)
                    class_labels = np.full(len(points), 255, dtype=int)
                else:
                    print("Segmentation successful", flush=True)
            except Exception as e:
                print(f"Segmentation error: {e}", flush=True)
                class_labels = np.full(len(points), 255, dtype=int)
        
        # Print class distribution
        unique_classes, counts = np.unique(class_labels, return_counts=True)
        for class_id, count in zip(unique_classes, counts):
            class_name = get_class_name(class_id)
            print(f"Class {class_id} ({class_name}): {count} points", flush=True)
        
        # Convert class labels to RGB colors
        class_colors = get_class_colors()
        rgb_colors = np.array([class_colors.get(label, [0.5, 0.5, 0.5]) for label in class_labels])
        rgb_colors = (rgb_colors * 255).astype(np.uint8)
        
        # Debug: Print some sample colors
        print(f"Sample RGB colors: {rgb_colors[:5]}", flush=True)
        print(f"RGB color range: min={rgb_colors.min()}, max={rgb_colors.max()}", flush=True)
        
        segmented_points = np.column_stack([
            points[:, 0], # x
            points[:, 1], # y
            points[:, 2], # z
            rgb_colors[:, 0],  # r
            rgb_colors[:, 1],  # g
            rgb_colors[:, 2],  # b
            class_labels       # class
        ])
        
        # Clear CUDA cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return segmented_points
    
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

def run_cylinder3d_segmentation(points, model):
    """Run Cylinder3D segmentation using WildScenes evaluation format"""
    try:
        # Preprocess points
        processed_points, preprocessed_mask = preprocess_points(points)
        
        if len(processed_points) == 0:
            return None
        
        # Process points with random sampling to avoid spiral patterns
        max_points = 200000  # Higher limit for better coverage
        if len(processed_points) > max_points:
            # Use random sampling to break spiral patterns
            indices = np.random.choice(len(processed_points), max_points, replace=False)
            processed_points = processed_points[indices]
            original_indices = indices
        else:
            original_indices = np.arange(len(processed_points))
            
        # Convert to tensor
        device = next(model.parameters()).device
        points_tensor = torch.from_numpy(processed_points).float().to(device)
        
        # Create data sample
        data_sample = Det3DDataSample()
        data_sample.set_metainfo({
            'sample_idx': 0,
            'lidar_points': {
                'lidar_path': 'ros2_frame',
                'num_pts_feats': 3
            }
        })
        
        inputs = {'points': [points_tensor]}
        data_samples = [data_sample]
        
        # Run inference
        try:
            processed_data = model.data_preprocessor({
                'inputs': inputs,
                'data_samples': data_samples
            })
            
            with torch.no_grad():
                results = model.predict(
                    processed_data['inputs'],
                    processed_data['data_samples']
                )
        except Exception as e:
            return None
        
        if results and len(results) > 0:
            pred_result = results[0]
            if hasattr(pred_result, 'pred_pts_seg') and pred_result.pred_pts_seg is not None:
                pred_data = pred_result.pred_pts_seg
                
                # Extract segmentation tensor
                if hasattr(pred_data, 'pts_semantic_mask'):
                    pred_tensor = pred_data.pts_semantic_mask
                else:
                    return None
                
                # Convert to numpy
                predictions = pred_tensor.cpu().numpy()
                if predictions.ndim > 1:
                    predictions = predictions.flatten()
                predictions = predictions.astype(np.int32)
                
                # Map back to original size
                full_predictions = np.full(len(points), 255, dtype=int)  # Start with all unsegmented
                
                # Get the indices of points that passed preprocessing
                preprocessed_indices = np.where(preprocessed_mask)[0]
                
                # Map predictions back to their original positions
                for i, pred in enumerate(predictions):
                    if i < len(original_indices):
                        # Map from processed index to original point cloud index
                        processed_idx = original_indices[i]
                        if processed_idx < len(preprocessed_indices):
                            original_idx = preprocessed_indices[processed_idx]
                            full_predictions[original_idx] = pred
                
                return full_predictions
        
        return None
        
    except Exception as e:
        return None


def preprocess_points(points):
    """Preprocess point cloud for Cylinder3D model"""
    # Expanded range to include more points
    point_cloud_range = [-50, -3.14159265359, -10, 50, 3.14159265359, 20]
    
    mask = (
        (points[:, 0] >= point_cloud_range[0]) &
        (points[:, 0] <= point_cloud_range[3]) &
        (points[:, 1] >= point_cloud_range[1]) &
        (points[:, 1] <= point_cloud_range[4]) &
        (points[:, 2] >= point_cloud_range[2]) &
        (points[:, 2] <= point_cloud_range[5])
    )
    
    # Debug: Print filtering statistics
    total_points = len(points)
    filtered_points = np.sum(mask)
    print(f"Point cloud filtering: {filtered_points}/{total_points} points kept ({filtered_points/total_points*100:.1f}%)", flush=True)
    
    return points[mask, :3], mask

def initialize_model():
    """Initialize the WildScenes model"""
    print("Initializing WildScenes model...", flush=True)
    model_path = '/veggie_drive/wildscenes/pretrained_models/cylinder3d_wildscenes.pth'
    config_path = '/veggie_drive/wildscenes/config.py'
    
    print(f"Model path: {model_path}", flush=True)
    print(f"Config path: {config_path}", flush=True)
    
    model = load_model(model_path, config_path)
    if model is not None:
        print("Model loaded successfully!", flush=True)
    else:
        print("Model loading failed!", flush=True)
    
    return model

