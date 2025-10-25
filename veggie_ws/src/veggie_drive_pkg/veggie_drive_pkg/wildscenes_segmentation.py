#!/usr/bin/env python3
"""
Inference script for WildScenes Cylinder3D
Will be referenced by the WildScenes ROS2 node to run inference
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch

def get_class_colors():
    """Map class IDs to RGB colors (0-1 floats)."""
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
    """Names for semantic classes."""
    return [
        "bush", "dirt", "fence", "grass", "gravel", "log", "mud",
        "object", "other-terrain", "rock", "structure", "tree-foliage", "tree-trunk"
    ]

def get_class_name(class_id):
    """Return human-readable label for class_id."""
    names = get_class_names()
    if class_id == 255:
        return "unsegmented"
    elif class_id < len(names):
        return names[class_id]
    else:
        return "unknown"


def _try_import(module_name: str):
    """Attempt to import a module, but don't raise if it fails.
    We do this so we can skip optional pieces that drag in scipy/sklearn.
    """
    try:
        __import__(module_name)
        print(f"[wildscenes] imported {module_name}", flush=True)
    except ModuleNotFoundError:
        print(f"[wildscenes] skipped missing {module_name}", flush=True)


def _safe_register_all_modules():
    """
    Minimal version of OpenMMLab's register_all_modules():
    - imports model components so they register with mmengine registries
    - intentionally DOES NOT import evaluation/metrics (those pull in sklearn/scipy)
    """
    import mmdet3d.utils.setup_env  # sets seeds / deterministic flags etc.

    # core model pieces
    _try_import("mmdet3d.models")
    _try_import("mmdet3d.models.data_preprocessors")
    _try_import("mmdet3d.models.backbones")
    _try_import("mmdet3d.models.necks")
    _try_import("mmdet3d.models.decode_heads")
    _try_import("mmdet3d.models.dense_heads")
    _try_import("mmdet3d.models.segmentors")
    _try_import("mmdet3d.models.voxel_encoders")
    _try_import("mmdet3d.models.middle_encoders")

    # some mmdet3d releases have this, some don't
    _try_import("mmdet3d.models.fusion_layers")

    # dataset pipeline bits (but NOT evaluation metrics)
    _try_import("mmdet3d.datasets")
    _try_import("mmdet3d.datasets.transforms")

    # NOTE: we intentionally do NOT import:
    #   mmdet3d.evaluation.metrics
    # because that drags in lyft_dataset_sdk -> sklearn -> scipy,
    # which is broken / heavy on Jetson


# Prime registries BEFORE importing MODELS / Det3DDataSample
_safe_register_all_modules()

# Now we can safely pull these in
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample


# Global singleton model instance
model = None


def load_model(model_path: str, config_path: str):
    """Load the Cylinder3D model using the provided checkpoint + config."""
    global model

    print(f"[wildscenes] Loading model from: {model_path}", flush=True)
    print(f"[wildscenes] Loading config from: {config_path}", flush=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        # clear CUDA cache if available
        if torch.cuda.is_available():
            print("[wildscenes] CUDA available, emptying cache first", flush=True)
            torch.cuda.empty_cache()
        else:
            print("[wildscenes] CUDA not available, using CPU", flush=True)

        print("[wildscenes] Loading config...", flush=True)
        cfg = Config.fromfile(config_path)
        print("[wildscenes] Config loaded successfully", flush=True)

        # ---- PATCH: fix data_preprocessor ----
        dp_block = cfg.model.get('data_preprocessor', None)
        if dp_block is not None:
            dp_type = dp_block.get('type', None)
            print(f"[wildscenes] original data_preprocessor type: {dp_type}", flush=True)

            # The config likely says type='Det3DDataPreprocessor', which isn't in
            # the registry here because we didn't call full register_all_modules().
            # That causes KeyError during MODELS.build().
            #
            # For inference, we can replace it with a very basic mmengine preprocessor.
        if dp_type == 'Det3DDataPreprocessor':
            print("[wildscenes] Replacing Det3DDataPreprocessor "
                "with BaseDataPreprocessor for runtime inference", flush=True)
            cfg.model['data_preprocessor'] = dict(
                type='BaseDataPreprocessor'
            )

        else:
            print("[wildscenes] config had no data_preprocessor block", flush=True)
        # ---- END PATCH ----

        print("[wildscenes] Building model...", flush=True)
        model = MODELS.build(cfg.model)
        print("[wildscenes] Model built successfully", flush=True)

        print("[wildscenes] Loading checkpoint...", flush=True)
        _ = load_checkpoint(model, model_path, map_location='cpu')
        print("[wildscenes] Checkpoint loaded", flush=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[wildscenes] Using device: {device}", flush=True)

        try:
            print("[wildscenes] Moving model to device + eval()", flush=True)
            model = model.to(device)
            model.eval()
            print("[wildscenes] Model moved to device and set to eval()", flush=True)

            if device.type == 'cuda':
                torch.cuda.empty_cache()
                print("[wildscenes] CUDA cache cleared after load", flush=True)

            print("[wildscenes] Model loading completed OK", flush=True)
            return model

        except RuntimeError as cuda_error:
            print(f"[wildscenes] CUDA error: {cuda_error}", flush=True)
            if "CUDA" in str(cuda_error):
                print("[wildscenes] Falling back to CPU", flush=True)
                device = torch.device("cpu")
                model = model.to(device)
                model.eval()
                print("[wildscenes] Model on CPU now", flush=True)
                return model
            else:
                raise cuda_error

    except Exception as e:
        print(f"[wildscenes] Error loading model: {e}", flush=True)
        return None


def run_segmentation(points: np.ndarray):
    """
    Run semantic segmentation on incoming point cloud.
    Returns Nx7 array: [x,y,z,r,g,b,class_id]
    """
    global model

    try:
        if model is None:
            print("[wildscenes] Model not loaded - returning all 'unsegmented'", flush=True)
            n_points = len(points)
            class_labels = np.full(n_points, 255, dtype=int)
        else:
            print("[wildscenes] Running Cylinder3D segmentation...", flush=True)
            try:
                class_labels = run_cylinder3d_segmentation(points, model)
                if class_labels is None:
                    print("[wildscenes] Segmentation failed, marking all unsegmented", flush=True)
                    class_labels = np.full(len(points), 255, dtype=int)
                else:
                    print("[wildscenes] Segmentation successful", flush=True)
            except Exception as e:
                print(f"[wildscenes] Segmentation error: {e}", flush=True)
                class_labels = np.full(len(points), 255, dtype=int)

        # Log distribution
        uniq, counts = np.unique(class_labels, return_counts=True)
        for cid, cnt in zip(uniq, counts):
            cname = get_class_name(cid)
            print(f"[wildscenes] class {cid} ({cname}): {cnt} pts", flush=True)

        # map to RGB (0-255 uint8)
        colors = get_class_colors()
        rgb = np.array([colors.get(lbl, [0.5, 0.5, 0.5]) for lbl in class_labels])
        rgb = (rgb * 255).astype(np.uint8)

        print(f"[wildscenes] sample RGB rows: {rgb[:5]}", flush=True)
        print(f"[wildscenes] rgb range: {rgb.min()}..{rgb.max()}", flush=True)

        segmented_points = np.column_stack([
            points[:, 0],  # x
            points[:, 1],  # y
            points[:, 2],  # z
            rgb[:, 0],     # r
            rgb[:, 1],     # g
            rgb[:, 2],     # b
            class_labels   # class id
        ])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return segmented_points

    except Exception as e:
        print(f"[wildscenes] run_segmentation fatal: {e}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None


def run_cylinder3d_segmentation(points: np.ndarray, model):
    """
    Internal helper to actually call the model.
    """
    try:
        processed_points, mask = preprocess_points(points)
        if len(processed_points) == 0:
            return None

        # limit points to keep runtime under control
        max_points = 200000
        if len(processed_points) > max_points:
            idx = np.random.choice(len(processed_points), max_points, replace=False)
            processed_points = processed_points[idx]
            original_indices = idx
        else:
            original_indices = np.arange(len(processed_points))

        device = next(model.parameters()).device
        pts_tensor = torch.from_numpy(processed_points).float().to(device)

        # wrap inputs in Det3DDataSample so predict() is happy
        data_sample = Det3DDataSample()
        data_sample.set_metainfo({
            'sample_idx': 0,
            'lidar_points': {
                'lidar_path': 'ros2_frame',
                'num_pts_feats': 3
            }
        })

        inputs = {'points': [pts_tensor]}
        data_samples = [data_sample]

        try:
            # model.data_preprocessor() prepares the batch dict
            processed_batch = model.data_preprocessor({
                'inputs': inputs,
                'data_samples': data_samples
            })

            with torch.no_grad():
                results = model.predict(
                    processed_batch['inputs'],
                    processed_batch['data_samples']
                )
        except Exception as e:
            print(f"[wildscenes] inference error: {e}", flush=True)
            return None

        if not results or len(results) == 0:
            return None

        pred_result = results[0]
        if not hasattr(pred_result, 'pred_pts_seg') or pred_result.pred_pts_seg is None:
            return None

        pred_data = pred_result.pred_pts_seg

        # mmseg-style point mask sometimes lives in pts_semantic_mask
        if hasattr(pred_data, 'pts_semantic_mask'):
            pred_tensor = pred_data.pts_semantic_mask
        else:
            print("[wildscenes] no pts_semantic_mask on pred_pts_seg", flush=True)
            return None

        pred_np = pred_tensor.cpu().numpy()
        if pred_np.ndim > 1:
            pred_np = pred_np.flatten()
        pred_np = pred_np.astype(np.int32)

        # map predictions back into full cloud
        full_pred = np.full(len(points), 255, dtype=int)  # default unsegmented
        kept_indices = np.where(mask)[0]

        for i, cls_id in enumerate(pred_np):
            if i < len(original_indices):
                proc_idx = original_indices[i]
                if proc_idx < len(kept_indices):
                    orig_idx = kept_indices[proc_idx]
                    full_pred[orig_idx] = cls_id

        return full_pred

    except Exception as e:
        print(f"[wildscenes] run_cylinder3d_segmentation fatal: {e}", flush=True)
        return None


def preprocess_points(points: np.ndarray):
    """
    Filter raw LiDAR into the region-of-interest the model expects.
    Returns (filtered_points_xyz, mask)
    """
    # match cylinder3d polar range-ish
    point_cloud_range = [-50, -3.14159265359, -10,
                          50,  3.14159265359,  20]

    mask = (
        (points[:, 0] >= point_cloud_range[0]) &
        (points[:, 0] <= point_cloud_range[3]) &
        (points[:, 1] >= point_cloud_range[1]) &
        (points[:, 1] <= point_cloud_range[4]) &
        (points[:, 2] >= point_cloud_range[2]) &
        (points[:, 2] <= point_cloud_range[5])
    )

    total = len(points)
    kept = int(mask.sum())
    pct = (kept / total * 100.0) if total > 0 else 0.0
    print(f"[wildscenes] point filter: {kept}/{total} kept ({pct:.1f}%)", flush=True)

    return points[mask, :3], mask


def initialize_model():
    """
    Called by the ROS2 node once at startup.
    """
    print("[wildscenes] Initializing WildScenes model...", flush=True)

    model_ckpt = "/veggie_drive/wildscenes/pretrained_models/cylinder3d_wildscenes.pth"
    config_file = "/veggie_drive/wildscenes/config.py"

    print(f"[wildscenes] model_ckpt: {model_ckpt}", flush=True)
    print(f"[wildscenes] config_file: {config_file}", flush=True)

    mdl = load_model(model_ckpt, config_file)
    if mdl is None:
        print("[wildscenes] Model loading FAILED", flush=True)
    else:
        print("[wildscenes] Model loaded OK", flush=True)

    return mdl
