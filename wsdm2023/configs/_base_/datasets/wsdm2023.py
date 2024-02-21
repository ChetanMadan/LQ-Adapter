# Copyright (c) OpenMMLab. All rights reserved.
# dataset settings
# dataset_type = 'WSDMCocoDataset'
dataset_type = 'CocoDatasetCustom'
# data_root = 'data/wsdm2023/'
data_root = '../detection/data/GBCU/'

classes = ('gb',)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadRefer', tag='question'),
    dict(type='RandomParaPhrase', 
         phrase_cache=data_root+'annotations/paraphrase_train.json'),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlipWithRefer', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='TokenizeRefer', max_sent_len=128),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'refer',
         'r_mask', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadRefer', tag='question'),
    dict(type='MultiScaleFlipAug',
         img_scale=(1333, 800),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlipWithRefer'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='TokenizeRefer', max_sent_len=128),
             dict(type='Collect', keys=['img', 'refer', 'r_mask']),
         ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(type=dataset_type,
               ann_file=data_root + 'new_splits/gb_train_4.json',
               img_prefix=data_root + 'imgs/',
               classes=classes,
               pipeline=train_pipeline),
    val=dict(type=dataset_type,
             ann_file=data_root + 'new_splits/gb_test_4.json',
             img_prefix=data_root + 'imgs/',
             classes=classes,
             pipeline=test_pipeline),
    test=dict(type=dataset_type,
              ann_file=data_root + 'new_splits/gb_test_4.json',
              img_prefix=data_root + 'imgs/',
              classes=classes,
              pipeline=test_pipeline))

evaluation = dict(metric=['bbox'])