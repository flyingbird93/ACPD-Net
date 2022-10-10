import copy
import os.path as osp
import os, json

import pandas as pd
import scipy.io as scio
import torch

import mmcv
from mmdet.apis import set_random_seed
import numpy as np
from PIL import Image
from test_cropping import parse_annotations, parse_FLMS_annotations, parse_CUHK_annotations
from test_cropping import evaluate_on_FLMS, evaluate_on_CUHK, evaluate_on_FCDB
from test_cropping import evaluate_on_FLMS_best_one, evaluate_on_FCDB_best_one

# modify config
from mmcv import Config
# Train a new detector
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.apis import init_detector, inference_detector
from mmdet.apis import show_result_pyplot


cfg = Config.fromfile('./configs/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco.py')


# Modify dataset type and path
cfg.dataset_type = 'FCDBDataset'
cfg.data_root = '/home/vr/Work/Cropping/dataset/FCDB/'

cfg.data.test.type = 'FCDBDataset'
cfg.data.test.data_root = '/home/vr/Work/Cropping/dataset/FCDB/'
cfg.data.test.ann_file = 'cropping_testing_set.json'
cfg.data.test.img_prefix = '/home/vr/Work/Cropping/dataset/FCDB/data'

cfg.data.val.type = 'FLMSDataset'
cfg.data.val.data_root = '/home/vr/Work/Cropping/dataset/FLMS/'
cfg.data.val.ann_file = '500_image_dataset.mat'
cfg.data.val.img_prefix = '/home/vr/Work/Cropping/dataset/FLMS/image'

cfg.data.train.type = 'FCDBDataset'
cfg.data.train.data_root = '/home/vr/Work/Cropping/dataset/FCDB/'
cfg.data.train.ann_file = 'cropping_training_set.json'
cfg.data.train.img_prefix = '/home/vr/Work/Cropping/dataset/FCDB/data/'

# modify num classes of the model in box head
cfg.model.roi_head.bbox_head.num_classes = 2
# cfg.model.bbox_head.num_classes = 2
# If we need to finetune a model based on a pre-trained detector, we need to
# use load_from to set the path of checkpoints.
cfg.load_from = 'checkpoints/faster_rcnn_r50_caffe_c4_mstrain_1x_coco_20220316_150527-db276fed.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './tutorial_exps'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 100
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 2

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]


# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')


# Build dataset
datasets = [build_dataset(cfg.data.train)]
test_datasets = [build_dataset(cfg.data.val)]
# FLMS_test_datsets = [build_dataset(cfg.data.val)]

test_anno_dict, img_w_list, img_h_list = parse_annotations(cfg.data.test.data_root, cfg.data.test.img_prefix)
test_FMLS_anno_dict, FLMS_img_w_list, FLMS_img_h_list = parse_FLMS_annotations(cfg.data.val.data_root, cfg.data.val.img_prefix)
# test_CUHK_anno_dict, CUHK_img_w_list, CUHK_img_h_list = parse_CUHK_annotations(cfg.data.test.data_root, cfg.data.test.img_prefix)
# print(test_FMLS_anno_dict)

# Build the detector
model = build_detector(cfg.model)
# print(model)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)


# evaluate FCDB test data
fcdb_test_ori_shape = []
fcdb_test_img_shape = []
test_img_list = []
test_pred_result_bbox = []
test_gt_result_bbox = []
file_path_list = []
model.cfg = cfg
img_root = '/home/vr/Work/Cropping/dataset/FCDB/data/'

# test fps
fps = 0
num_warmup = 5
pure_inf_time = 0
max_iter = 300
log_interval = 50


for i, img_path in enumerate(test_anno_dict.keys()):
    # print(img_path)
    img = img_root + img_path
    result, img_ori_shape, img_test_shape = inference_detector(model, img) #, i, num_warmup, pure_inf_time)

    fcdb_test_ori_shape.append(img_ori_shape)
    fcdb_test_img_shape.append(img_test_shape)
    file_path_list.append(img_path)
    test_pred_result_bbox.append(result)
    test_gt_result_bbox.append(test_anno_dict[img_path])
    # show_result_pyplot(model, img, result)

# print("test fps: ", fps)
# print('FCDB test predict num: ', len(test_pred_result_bbox[0]))
# print(test_pred_result_bbox[0])

# evaluate FLMS test data
FLMS_test_ori_shape = []
FLMS_test_img_shape = []
FLMS_test_img_list = []
FLMS_test_pred_result_bbox = []
FLMS_test_gt_result_bbox = []
FLMS_file_path_list = []
model.cfg = cfg
FLMS_img_root = '/home/vr/Work/Cropping/dataset/FLMS/image/'

for i in test_FMLS_anno_dict.keys():
    img = FLMS_img_root + i
    result, img_ori_shape, img_test_shape = inference_detector(model, img)
    # print(result)
    FLMS_test_ori_shape.append(img_ori_shape)
    FLMS_test_img_shape.append(img_test_shape)
    FLMS_file_path_list.append(i)
    FLMS_test_pred_result_bbox.append(result)  # 500 [list 2, (rand, 5) (rand, 5)]
    FLMS_test_gt_result_bbox.append(test_FMLS_anno_dict[i]) # 500, 10, 4


# Evaluate
# shape (h, w, 3)
# img_w_list, img_h_list
# FLMS_img_w_list, FLMS_img_h_list
_, _, all_iou_list, all_file_path = evaluate_on_FCDB(test_pred_result_bbox, torch.Tensor(test_gt_result_bbox), img_w_list, img_h_list, file_path_list, img_root)
evaluate_on_FLMS(FLMS_test_pred_result_bbox, FLMS_test_gt_result_bbox, FLMS_img_w_list, FLMS_img_h_list, FLMS_file_path_list, FLMS_img_root)

