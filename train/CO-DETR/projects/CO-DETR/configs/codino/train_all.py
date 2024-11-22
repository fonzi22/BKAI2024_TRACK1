import sys
sys.path.append('/home/s48gb/Desktop/GenAI4E/test_OD/BKAI2024_TRACK1/train/CO-DETR/')
_base_ = ['co_dino_5scale_r50_8xb2_1x_coco.py']

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'  # noqa
# load_from = '/home/s48gb/Desktop/GenAI4E/test_OD/BKAI2024_TRACK1/train/CO-DETR/work_dirs/train_all/codetr_augment_1e.pth'
# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536]),
    query_head=dict(
        dn_cfg=dict(box_noise_scale=0.4, group_cfg=dict(num_dn_queries=500)),
        transformer=dict(encoder=dict(with_cp=6))))

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(
#         type='RandomChoice',
#         transforms=[
#             [
#                 dict(
#                     type='RandomChoiceResize',
#                     scales=[(480, 2048), (512, 2048), (544, 2048), (576, 2048),
#                             (608, 2048), (640, 2048), (672, 2048), (704, 2048),
#                             (736, 2048), (768, 2048), (800, 2048), (832, 2048),
#                             (864, 2048), (896, 2048), (928, 2048), (960, 2048),
#                             (992, 2048), (1024, 2048), (1056, 2048),
#                             (1088, 2048), (1120, 2048), (1152, 2048),
#                             (1184, 2048), (1216, 2048), (1248, 2048),
#                             (1280, 2048), (1312, 2048), (1344, 2048),
#                             (1376, 2048), (1408, 2048), (1440, 2048),
#                             (1472, 2048), (1504, 2048), (1536, 2048)],
#                     keep_ratio=True)
#             ],
#             [
#                 dict(
#                     type='RandomChoiceResize',
#                     # The radio of all image in train dataset < 7
#                     # follow the original implement
#                     scales=[(400, 4200), (500, 4200), (600, 4200)],
#                     keep_ratio=True),
#                 dict(
#                     type='RandomCrop',
#                     crop_type='absolute_range',
#                     crop_size=(384, 600),
#                     allow_negative_crop=True),
#                 dict(
#                     type='RandomChoiceResize',
#                     scales=[(480, 2048), (512, 2048), (544, 2048), (576, 2048),
#                             (608, 2048), (640, 2048), (672, 2048), (704, 2048),
#                             (736, 2048), (768, 2048), (800, 2048), (832, 2048),
#                             (864, 2048), (896, 2048), (928, 2048), (960, 2048),
#                             (992, 2048), (1024, 2048), (1056, 2048),
#                             (1088, 2048), (1120, 2048), (1152, 2048),
#                             (1184, 2048), (1216, 2048), (1248, 2048),
#                             (1280, 2048), (1312, 2048), (1344, 2048),
#                             (1376, 2048), (1408, 2048), (1440, 2048),
#                             (1472, 2048), (1504, 2048), (1536, 2048)],
#                     keep_ratio=True)
#             ]
#         ]),
#     dict(type='PackDetInputs')
# ]

# data_root = '/Data/Visdrone_Fisheye8K/'
# metainfo = {
#     'classes': ('Bike', 'Car', 'Bus', 'Truck'),
# }
# dataset_type = 'CocoDataset'

# train_dataloader = dict(
#     batch_size=2, num_workers=1,
#     dataset=dict(
#         type=dataset_type,
#         pipeline=train_pipeline,
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file='/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train_augment.json',
#         data_prefix=dict(img='/home/s48gb/Desktop/GenAI4E/test_OD/dataset/')
#     ))

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', scale=(2048, 1280), keep_ratio=True),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]

# val_dataloader = dict(
#     dataset=dict(
#         type=dataset_type,
#         pipeline=test_pipeline,
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file='/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train_eval.json',
#         data_prefix=dict(img='/home/s48gb/Desktop/GenAI4E/test_OD/dataset/')
#         ))
# test_dataloader = val_dataloader

# val_evaluator = dict(  # Validation evaluator config
#     type='CocoMetric',  # The coco metric used to evaluate AR, AP, and mAP for detection and instance segmentation
#     ann_file='/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train_eval.json',  # Annotation file path
#     metric=['bbox'],  # Metrics to be evaluated, `bbox` for detection and `segm` for instance segmentation
#     format_only=False)
# test_evaluator = val_evaluator  # Testing evaluator config

# optim_wrapper = dict(optimizer=dict(lr=1e-4))

# max_epochs = 4
# train_cfg = dict(max_epochs=max_epochs)

# param_scheduler = [
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=max_epochs,
#         by_epoch=True,
#         milestones=[8],
#         gamma=0.1)
# ]
data_root = '/Data/dataset/'  # Correct dataset location inside the container
metainfo = {
    'classes': ('Bike', 'Car', 'Bus', 'Truck'),
}
dataset_type = 'CocoDataset'

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=data_root + 'train_augment.json',  # Adjust to your actual annotation file
        data_prefix=dict(img=data_root)
    ))


val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=data_root + 'train_eval.json',
        data_prefix=dict(img=data_root)
    ))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'train_eval.json',
    metric=['bbox'],
    format_only=False
)
test_evaluator = val_evaluator  # Testing evaluator config

# # Disable validation
# val_dataloader = None
# val_cfg = None
# val_evaluator = None

# test_dataloader = None  # If you also want to skip test evaluation
# test_cfg = None
# test_evaluator = None


optim_wrapper = dict(optimizer=dict(lr=1e-4))

max_epochs = 2
train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30],
        gamma=0.1)
]

