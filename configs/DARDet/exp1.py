# model settings
model = dict(
    type='DARDet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='DARDetHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=True,#False
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
            loss_rbox=dict( type='RotatedIoULoss', loss_weight=1.5),
            loss_rbox_refine=dict( type='RotatedIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        # nms=dict(type='nms', iou_threshold=0.1),
        max_per_img=1500))

# data setting
dataset_type = 'PODDataset'
data_root = './datasets/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
classes = ('table', 'keyvalue')
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True,poly2mask=False),
    dict(type='Rotate', prob=0.9, auto_bound=True, max_rotate_angle=360),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale = [(320,1024), (416,1024), (512,1024), (608,1024), (704,1024), (800,1024)],
                # img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                #            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                #            (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale = [(416,1024), (512,1024), (608,1024)],
                        #   img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                          multiscale_mode='value',
                          keep_ratio=True),
                    #   dict(
                    #       type='RandomCrop',
                    #       crop_type='absolute_range',
                    #       crop_size=(384, 600),
                    #       allow_negative_crop=True),
                    #   dict(
                    #       type='Resize',
                    #       img_scale = [(320,1024), (416,1024), (512,1024), (608,1024), (704,1024), (800,1024)],
                        #   img_scale=[(480, 1333), (512, 1333), (544, 1333),
                        #              (576, 1333), (608, 1333), (640, 1333),
                        #              (672, 1333), (704, 1333), (736, 1333),
                        #              (768, 1333), (800, 1333)],
                        #   multiscale_mode='value',
                        #   override=True,
                        #   keep_ratio=True)
                  ]]),
    # dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    # dict(type='RandomFlip', direction=['horizontal','vertical', 'diagonal'], flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'], meta_keys = ('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg')),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 1024),
        flip=False,
        transforms=[
            dict(type='Resize',img_scale=(512,1024), keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'], meta_keys = ('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg')),
        ])
]
train_dataset = ("POD_RevA", "knowledge_lake_pod_part1", "knowledge_lake_pod_part2", "financial_report_pod",)
test_dataset = ("rotated_360_POD_RevB_combined","rotated_45_POD_RevB_combined","POD_RevB_combined",)
dataset_1 = dict(
    classes = classes,
    type = dataset_type,
    ann_file = data_root + 'POD_RevA/train.json',
    img_prefix = data_root + 'POD_RevA/JPEGImages',
    pipeline = train_pipeline
)
dataset_2 = dict(
    classes = classes,
    type = dataset_type,
    ann_file = data_root + 'knowledge_lake_pod_part1/train.json',
    img_prefix = data_root + 'knowledge_lake_pod_part1/JPEGImages',
    pipeline = train_pipeline
)
dataset_3 = dict(
    classes = classes,
    type = dataset_type,
    ann_file = data_root + 'knowledge_lake_pod_part2/train.json',
    img_prefix = data_root + 'knowledge_lake_pod_part2/JPEGImages',
    pipeline = train_pipeline
)
dataset_4 = dict(
    classes = classes,
    type = dataset_type,
    ann_file = data_root + 'financial_report_pod/train.json',
    img_prefix = data_root + 'financial_report_pod/JPEGImages',
    pipeline = train_pipeline
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        classes=classes,
        type='ConcatDataset',
        datasets=[dataset_1,dataset_2,dataset_3,dataset_4]),
    val=dict(
        classes = classes,
        type = dataset_type,
        ann_file = data_root + 'knowledge_lake_pod_part2/train.json',
        img_prefix = data_root + 'knowledge_lake_pod_part2/JPEGImages',
        pipeline = test_pipeline
    ),
    # val=dict(
    #     classes=classes,
    #     type=dataset_type,
    #     samples_per_gpu=6,
    #     ann_file=data_root + 'annotations/train_3.json',
    #     img_prefix=data_root + 'val/',
    #     pipeline=test_pipeline),
    test=dict(
        classes=classes,
        type=dataset_type,
        pipeline=test_pipeline)
    )

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[32, 44])
runner = dict(type='EpochBasedRunner', max_epochs=50)
work_dir = '/datadisk/v-jiaweiwang/DARDet'
load_from =None#'/media/zf/E/mmdetection213_2080/checkpoint/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-6879c318.pth'
resume_from = None #'/media/zf/E/Dataset/dota1-split-1024/workdir/DARDet_r50_DCN_rotate/latest.pth'
evaluation = dict(interval=20, metric='bbox',
        gt_dir='/datadisk/v-jiaweiwang/data/POD_RevB_combined/xml')
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
