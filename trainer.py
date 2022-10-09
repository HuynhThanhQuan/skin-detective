# Basic import
import pandas as pd
import numpy as np
import cv2
import math
import os
import re
import matplotlib.pyplot as plt
import sys
import argparse
import shutil
import subprocess
from copy import deepcopy
import logging
import datetime
from pathlib import Path
import json

# Import image, DL packages
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader
import tensorflow as tf

# Misc
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

# Import package file
from engine import train_one_epoch, evaluate
import utils
import logging
import acne_configs
from acne_dataset import AcneDataset
from acne_utils import get_train_transform, get_test_transform, collate_fn


logging.debug(f"PyTorch Version: {torch.__version__}")
logging.debug(f"Torchvision Version: {torchvision.__version__}")

    
def read_dataset(args, dataset_path):
    # Check
    ds_loc = Path(dataset_path)
    train_ds_loc = ds_loc / 'train'
    val_ds_loc = ds_loc / 'val'
    test_ds_loc = ds_loc / 'test'
    assert train_ds_loc.exists(), 'Not found training dataset'
    assert val_ds_loc.exists(), 'Not found val dataset'
    assert test_ds_loc.exists(), 'Not found test dataset'
    
    train_img = train_ds_loc / 'image'
    train_bbox = train_ds_loc / 'coco_instances.json'
    assert train_img.exists(), 'Not found training image'
    assert train_bbox.exists(), 'Not found training bounding box'
    val_img = val_ds_loc / 'image'
    val_bbox = val_ds_loc / 'coco_instances.json'
    assert val_img.exists(), 'Not found val image'
    assert val_bbox.exists(), 'Not found val bounding box'
    test_img = test_ds_loc / 'image'
    test_bbox = test_ds_loc / 'coco_instances.json'
    assert test_img.exists(), 'Not found test image'
    assert test_bbox.exists(), 'Not found test bounding box'
    
    # Apply transformation
    train_dataset = AcneDataset(train_ds_loc, get_train_transform())
    val_dataset = AcneDataset(val_ds_loc, get_test_transform())
    test_dataset = AcneDataset(test_ds_loc, get_test_transform())
    # Dataloader
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    return train_data_loader, val_data_loader, test_data_loader
    
    
def reload_retrained_models(args):
    cache_models = [i for i in os.listdir(args.model_store) if '.pkl' in i]
    cache_index = [int(re.findall(r'_epoch(\d+)', i)[0]) for i in cache_models]
    if len(cache_index) > 0:
        max_idx = max(cache_index)
    else:
        max_idx = 0
    return max_idx, cache_index
    
    
def run(args):
    train_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, 'tf', 'train'))
    test_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, 'tf', 'test'))
    
    s_time = datetime.datetime.now()
    
    train_data_loader, val_data_loader, test_data_loader = read_dataset(args, args.data)
    
    # GPU selection
    cpu = torch.device('cpu')
    device = torch.device("cuda:%s" % args.cuda if torch.cuda.is_available() else "cpu")
    logging.info(f'Selected GPU: {device}')

    # Prepare training
    NUM_CLASSES = len(acne_configs.ACNE_ID)
    logging.info(f'Number of classes: {NUM_CLASSES}')
    logging.info('===============TRAINING=============')
    start_epoch = 0
    max_cache_id, cache_index = reload_retrained_models(args)

    # Continue training or new training
    if (max_cache_id!=0) and (args.resume is True):
        logging.info('Continuous training...')
        logging.info(f'Found {len(cache_index)} cached models, maximum index model: {max_cache_id}')
        last_model_fn = f'{args.model_store}/{args.backbone}_epoch{max_cache_id}.pkl'
        model = torch.load(last_model_fn, map_location='cuda:%s' % args.cuda)
        model = model.train()
        start_epoch = max_cache_id + 1
    else:
        # 1. Use pretrained model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

        # 2. Modified with different backbones
    #     backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    #     backbone.out_channels = 1280
    #     anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                    aspect_ratios=((0.5, 1.0, 2.0),))
    #     roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=7,
    #                                                 sampling_ratio=2)
    #     model = FasterRCNN(backbone,
    #                    num_classes=NUM_CLASSES,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)


    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    
    if args.optimizer=='sgd':
        optimizer = torch.optim.SGD(params, lr=args.learning_rate)
    elif args.optimizer=='adamw':
        optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    else:
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)
        
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=args.print_freq, writer=train_summary_writer)
        lr_scheduler.step()
        evaluate(model, val_data_loader, device=device, writer=test_summary_writer)
        if epoch % args.save_epoch == (args.save_epoch - 1):
            lastest_model_fn = f'{args.model_store}/{args.backbone}_epoch{epoch}.pkl'
            torch.save(model, lastest_model_fn)
            
    logging.info('Evaluating test dataset')
    evaluate(model, test_data_loader, device=device, writer=test_summary_writer)
    
    e_time = datetime.datetime.now() - s_time
    logging.info(f'Training completed in {e_time}')