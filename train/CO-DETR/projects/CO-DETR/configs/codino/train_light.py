import sys

_base_ = ['co_dino_5scale_r50_8xb2_1x_coco.py']

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_lsj_r50_3x_coco-fe5a6829.pth'
# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_r50_1x_coco-7481f903.pth'
# load_from = '/home/s48gb/Desktop/GenAI4E/test_OD/AICITY2024_Track4/train/CO-DETR/work_dirs/train_light/best_coco_bbox_mAP_epoch_2.pth'

# model = dict(
#     use_lsj=False, data_preprocessor=dict(pad_mask=False, batch_augments=None))

# # train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# # from the default setting in mmdet.
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
#         ann_file='/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train_test.json',
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

# max_epochs = 2
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

data_root = '/Data/Visdrone_Fisheye8K/'
metainfo = {
    'classes': ('Bike', 'Car', 'Bus', 'Truck'),
}
dataset_type = 'CocoDataset'

train_dataloader = dict(
    batch_size=3, num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train_original.json',
        data_prefix=dict(img='/home/s48gb/Desktop/GenAI4E/test_OD/dataset/')
    ))


val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train_eval.json',
        data_prefix=dict(img='/home/s48gb/Desktop/GenAI4E/test_OD/dataset/')
        ))
test_dataloader = val_dataloader

val_evaluator = dict(  # Validation evaluator config
    type='CocoMetric',  # The coco metric used to evaluate AR, AP, and mAP for detection and instance segmentation
    ann_file='/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train_eval.json',  # Annotation file path
    metric=['bbox'],  # Metrics to be evaluated, `bbox` for detection and `segm` for instance segmentation
    format_only=False)
test_evaluator = val_evaluator  # Testing evaluator config

optim_wrapper = dict(optimizer=dict(lr=1e-4))

max_epochs = 4
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
