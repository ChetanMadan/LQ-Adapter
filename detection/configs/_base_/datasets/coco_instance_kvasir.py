# Copyright (c) OpenMMLab. All rights reserved.
# dataset settings

# dataset_type = 'CocoDataset'
# data_root = 'data/coco/'

dataset_type = 'CocoDatasetCustom'
# data_root = 'data/ddsm_updated/'
data_root = 'data/kvasir/'
# data_root = 'data/DDSM_actual/'

classes = ('mal',)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(330, 330),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=False),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    batch_size=2,
    train=dict(type=dataset_type,
               ann_file=data_root + 'train_new.json',
               img_prefix=data_root + 'imgs',
               classes=classes,
               pipeline=train_pipeline),
    val=dict(type=dataset_type,
             ann_file=data_root + 'test_new.json',
             img_prefix=data_root + 'imgs',
             classes=classes,
             pipeline=test_pipeline),
    test=dict(type=dataset_type,
              ann_file=data_root + 'test_new.json',
              img_prefix=data_root + 'imgs',
              classes=classes,
              pipeline=test_pipeline))


evaluation = dict(metric=['bbox'])
