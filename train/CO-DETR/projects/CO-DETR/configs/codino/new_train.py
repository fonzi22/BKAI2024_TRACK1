import sys

_base_ = ['co_dino_5scale_r50_8xb2_1x_coco.py']

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_lsj_r50_3x_coco-fe5a6829.pth'

# model = dict(
#     use_lsj=False, data_preprocessor=dict(pad_mask=False, batch_augments=None))

data_root = '/Data/data/'
metainfo = {
    'classes': ('Bike', 'Car', 'Bus', 'Truck'),
}
dataset_type = 'CocoDataset'

train_dataloader = dict(
    batch_size=2, num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train_test.json',
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
