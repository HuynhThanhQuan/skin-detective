#!/usr/bin/env python
# coding: utf-8

import torch, detectron2
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

import os
import pandas 
import numpy
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path('/home/ec2-user/skin-detective/data/final/f_circle/ds/')

dataset = ['train','val','test']

for ds in dataset:
    ds_loc = DATA / ds / 'image'
    register_coco_instances(f'acne_{ds}', {}, str(DATA / ds / 'coco_instances.json'), str(ds_loc))

acne_ds = DatasetCatalog.get('acne_train')

metadata = MetadataCatalog.get('acne_train')

for d in acne_ds[:2]:
    img = cv2.imread(d["file_name"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    visualizer = Visualizer(img[:, :, ::-1], scale=1)
    out = visualizer.draw_dataset_dict(d)
    plt.figure(figsize=(10,10))
    plt.imshow(out.get_image()[:, :, ::-1])


# ## TRAIN

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("acne_train",)
cfg.DATASETS.TEST = ("acne_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  
cfg.SOLVER.IMS_PER_BATCH = 2  
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300000    
cfg.SOLVER.STEPS = []        
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=True)
trainer.train()
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   
predictor = DefaultPredictor(cfg);

from detectron2.utils.visualizer import ColorMode
dataset_dicts = DatasetCatalog.get('acne_val')
for d in random.sample(dataset_dicts, 2):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im) 
    v = Visualizer(im[:, :, ::-1],
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator("acne_val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "acne_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))

evaluator = COCOEvaluator("acne_test", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "acne_test")
print(inference_on_dataset(predictor.model, val_loader, evaluator))