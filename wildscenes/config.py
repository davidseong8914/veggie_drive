_base_ = [
    '../_base_/datasets/wildscenes.py', '../_base_/models/cylinder3d.py',
    '../_base_/default_runtime.py'
]

# optimizer
lr = 0.001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[30],
        gamma=0.1)
]

train_dataloader = dict(batch_size=10, ) # can only use a batch size of 10 for Cylinder3D

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='miou', max_keep_ckpts=2),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook')
    # Remove visualization hook to avoid compatibility issues
)

grid_shape = [480, 360, 58] # 100%

model = dict(
    type='Cylinder3D',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=grid_shape,
            point_cloud_range=[0, -3.14159265359, -4, 50, 3.14159265359, 10],
            max_num_points=-1,
            max_voxels=-1,
        ),
    ),
    voxel_encoder=dict(
        type='SegVFE',
        feat_channels=[64, 128, 256, 256],
        in_channels=5,
        with_voxel_center=True,
        feat_compression=16,
        return_point_feats=False),
    backbone=dict(
        type='Asymm3DSpconv',
        grid_size=grid_shape,
        input_channels=16,
        base_channels=32,
        norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1)),
    decode_head=dict(
        type='Cylinder3DHead',
        channels=128,
        num_classes=13,
        loss_ce=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0,
            ignore_index=255),
        loss_lovasz=dict(type='LovaszLoss', loss_weight=1.0, reduction='none'),
    ),
    train_cfg=None,
    test_cfg=dict(mode='whole'),
)

# Custom dataset configuration overrides
data_root = 'data/processed/custom_dataset'
backend_args = None

class_names = (
    "bush", # 0
    "dirt", # 1
    "fence", # 2
    "grass", # 3
    "gravel", # 4
    "log", # 5
    "mud", # 6
    "object", # 7
    "other-terrain", # 8
    "rock", # 9
    "structure", # 10
    "tree-foliage", # 11
    "tree-trunk", # 12
)

# Test pipeline WITH labels for evaluation
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=3,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint16',  # Changed from np.int32 to np.uint16
        seg_offset=0,              # Set to 0 since we're storing raw labels
        dataset_type='wildscenes',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

# Override the test dataloader to use your custom data
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='WildScenesDataset3d',
        data_root=data_root,
        ann_file='custom_infos_test.pkl',
        pipeline=test_pipeline,
        metainfo=dict(
            classes=class_names,
            seg_label_mapping={
                0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
                10: 255, 11: 10, 12: 11, 13: 12, 14: 255, 255: 255
            },
            max_label=255
        ),
        modality=dict(use_lidar=True, use_camera=False),
        ignore_index=255,
        test_mode=True,
        backend_args=backend_args))

# Validation dataloader (copy of test for consistency)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='WildScenesDataset3d',
        data_root=data_root,
        ann_file='custom_infos_test.pkl',
        pipeline=test_pipeline,
        metainfo=dict(
            classes=class_names,
            seg_label_mapping={
                0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
                10: 255, 11: 10, 12: 11, 13: 12, 14: 255, 255: 255
            },
            max_label=255
        ),
        modality=dict(use_lidar=True, use_camera=False),
        ignore_index=255,
        test_mode=True,
        backend_args=backend_args))

# Standard evaluators - will work with dummy labels
val_evaluator = dict(type='SegMetric')
test_evaluator = dict(type='SegMetric')

# Visualization settings
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer')

# Random seed settings
randomness = dict(seed=0, deterministic=False, diff_rank_seed=True)

test_cfg = dict(
    type='TestLoop',
    # Save predictions to files
    save_predictions=True,
    prediction_save_path='results/predictions'
)