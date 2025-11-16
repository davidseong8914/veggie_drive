#!/usr/bin/env python3
"""
Inference script for WildScenes Cylinder3D
Used by the WildScenes ROS2 node to run inference
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
    """Attempt to import a module, but don't raise if it fails."""
    try:
        __import__(module_name)
        print(f"[wildscenes] imported {module_name}", flush=True)
    except ModuleNotFoundError:
        print(f"[wildscenes] skipped missing {module_name}", flush=True)


def _safe_register_all_modules():
    """
    Bring in core mmdet3d modules so registries are populated.
    We skip evaluation metrics to avoid sklearn/scipy.
    """
    import mmdet3d.utils.setup_env  # sets seeds / deterministic flags etc.

    _try_import("mmdet3d.models")
    _try_import("mmdet3d.models.data_preprocessors")
    _try_import("mmdet3d.models.backbones")
    _try_import("mmdet3d.models.necks")
    _try_import("mmdet3d.models.decode_heads")
    _try_import("mmdet3d.models.dense_heads")
    _try_import("mmdet3d.models.segmentors")
    _try_import("mmdet3d.models.voxel_encoders")
    _try_import("mmdet3d.models.middle_encoders")
    _try_import("mmdet3d.models.fusion_layers")

    _try_import("mmdet3d.datasets")
    _try_import("mmdet3d.datasets.transforms")


# prime registries
_safe_register_all_modules()

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample

# here's the critical import: Det3DDataPreprocessor class itself
try:
    from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
except Exception as e:
    # If this fails, we won't be able to voxelize. We'll warn loudly.
    Det3DDataPreprocessor = None
    print(f"[wildscenes] WARNING: could not import Det3DDataPreprocessor: {e}", flush=True)


# Global singleton
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

        # grab original preprocessor spec from config
        original_dp_cfg = cfg.model.get('data_preprocessor', None)
        print(f"[wildscenes] original data_preprocessor block: {original_dp_cfg}", flush=True)

        # safety check
        if original_dp_cfg is None:
            print("[wildscenes] WARNING: no data_preprocessor in cfg.model", flush=True)

        # We are going to TEMPORARILY replace it with a simple placeholder
        # just so MODELS.build() doesn't choke on registry lookup.
        cfg.model['data_preprocessor'] = dict(type='BaseDataPreprocessor')

        print("[wildscenes] Building model (with temporary BaseDataPreprocessor)...", flush=True)
        model_built = MODELS.build(cfg.model)
        print("[wildscenes] Model built successfully", flush=True)

        print("[wildscenes] Loading checkpoint...", flush=True)
        _ = load_checkpoint(model_built, model_path, map_location='cpu')
        print("[wildscenes] Checkpoint loaded", flush=True)

        # NOW: restore the real Det3DDataPreprocessor if we have it
        if (original_dp_cfg is not None
                and Det3DDataPreprocessor is not None
                and original_dp_cfg.get('type', '') == 'Det3DDataPreprocessor'):

            print("[wildscenes] Reconstructing Det3DDataPreprocessor manually...", flush=True)

            # pull params out
            # example original_dp_cfg:
            # {
            #   'type': 'Det3DDataPreprocessor',
            #   'voxel': True,
            #   'voxel_type': 'cylindrical',
            #   'voxel_layer': {
            #       'grid_shape': [...],
            #       'point_cloud_range': [...],
            #       'max_num_points': -1,
            #       'max_voxels': -1
            #   }
            # }
            dp_args = dict(original_dp_cfg)
            dp_args.pop('type', None)  # remove the 'type' key so we can pass kwargs

            try:
                real_dp = Det3DDataPreprocessor(**dp_args)
                model_built.data_preprocessor = real_dp
                print("[wildscenes] Attached Det3DDataPreprocessor to model.", flush=True)
            except Exception as e:
                print(f"[wildscenes] ERROR constructing Det3DDataPreprocessor: {e}", flush=True)
                print("[wildscenes] Keeping placeholder BaseDataPreprocessor. "
                      "Inference may still fail with 'voxels' later.", flush=True)

        else:
            print("[wildscenes] Could not attach Det3DDataPreprocessor (missing class or cfg).", flush=True)
            print("[wildscenes] We'll keep BaseDataPreprocessor which may cause 'voxels' KeyError.", flush=True)

        # move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[wildscenes] Using device: {device}", flush=True)
        try:
            print("[wildscenes] Moving model to device + eval()", flush=True)
            model_built = model_built.to(device)
            model_built.eval()
            print("[wildscenes] Model moved to device and set to eval()", flush=True)

            if device.type == 'cuda':
                torch.cuda.empty_cache()
                print("[wildscenes] CUDA cache cleared after load", flush=True)

        except RuntimeError as cuda_error:
            print(f"[wildscenes] CUDA error during .to(device): {cuda_error}", flush=True)
            if "CUDA" in str(cuda_error):
                print("[wildscenes] Falling back to CPU", flush=True)
                device = torch.device("cpu")
                model_built = model_built.to(device)
                model_built.eval()
                print("[wildscenes] Model on CPU now", flush=True)
            else:
                raise cuda_error

        # final assign
        model = model_built

        # sanity dump
        try:
            print(f"[wildscenes] Final data_preprocessor class: {type(model.data_preprocessor)}", flush=True)
        except Exception as e:
            print(f"[wildscenes] Couldn't inspect data_preprocessor: {e}", flush=True)

        print("[wildscenes] Model loading completed OK", flush=True)
        return model

    except Exception as e:
        print(f"[wildscenes] Error loading model: {e}", flush=True)
        model = None
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

        # Log label distribution
        uniq, counts = np.unique(class_labels, return_counts=True)
        for cid, cnt in zip(uniq, counts):
            cname = get_class_name(cid)
            print(f"[wildscenes] class {cid} ({cname}): {cnt} pts", flush=True)

        # map classes -> RGB (uint8)
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
    Actually call the model. Assumes model.data_preprocessor is now a working
    Det3DDataPreprocessor instance that knows how to voxelize.
    """
    try:
        processed_points, mask = preprocess_points(points)
        if len(processed_points) == 0:
            return None

        # limit runtime
        max_points = 200000
        if len(processed_points) > max_points:
            idx = np.random.choice(len(processed_points), max_points, replace=False)
            processed_points = processed_points[idx]
            original_indices = idx
        else:
            original_indices = np.arange(len(processed_points))

        device = next(model.parameters()).device
        pts_tensor = torch.from_numpy(processed_points).float().to(device)

        # minimal Det3DDataSample
        data_sample = Det3DDataSample()
        data_sample.set_metainfo({
            'sample_idx': 0,
            'lidar_points': {
                'lidar_path': 'ros2_frame',
                'num_pts_feats': 3
            }
        })

        # raw batch structure expected by preprocessor
        inputs = {'points': [pts_tensor]}
        data_samples = [data_sample]

        # let preprocessor voxelize / pack
        processed_batch = model.data_preprocessor({
            'inputs': inputs,
            'data_samples': data_samples
        })

        with torch.no_grad():
            results = model.predict(
                processed_batch['inputs'],
                processed_batch['data_samples']
            )

        if not results or len(results) == 0:
            print("[wildscenes] model.predict() returned empty results", flush=True)
            return None

        pred_result = results[0]
        if not hasattr(pred_result, 'pred_pts_seg') or pred_result.pred_pts_seg is None:
            print("[wildscenes] No pred_pts_seg in results[0]", flush=True)
            return None

        pred_data = pred_result.pred_pts_seg

        if hasattr(pred_data, 'pts_semantic_mask'):
            pred_tensor = pred_data.pts_semantic_mask
        else:
            print("[wildscenes] pred_pts_seg has no pts_semantic_mask", flush=True)
            return None

        pred_np = pred_tensor.cpu().numpy()
        if pred_np.ndim > 1:
            pred_np = pred_np.flatten()
        pred_np = pred_np.astype(np.int32)

        # Re-map predictions back to all original points
        full_pred = np.full(len(points), 255, dtype=int)
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
    Filter raw LiDAR points to model ROI.
    NOTE: y is being clamped to [-pi, pi] meters here, which is ~6.28 m wide,
    not truly cylindrical. We'll refine later; it's fine for now.
    """
    point_cloud_range = [
        -50,   -3.14159265359, -5,
         50,     3.14159265359, 15
    ]

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

    # print(f"[wildscenes] model_ckpt: {model_ckpt}", flush=True)
    # print(f"[wildscenes] config_file: {config_file}", flush=True)

    mdl = load_model(model_ckpt, config_file)
    if mdl is None:
        print("[wildscenes] Model loading FAILED", flush=True)
    else:
        print("[wildscenes] Model loaded OK", flush=True)

    return mdl
