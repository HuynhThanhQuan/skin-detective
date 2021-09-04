import pandas as pd
import numpy as np
import cv2
import math
import os
import re
import warnings
from copy import deepcopy
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
warnings.filterwarnings("ignore")
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import re
import xml.etree.ElementTree as ET
import os
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import copy
import sys
import argparse
import shutil
import subprocess


log_message = []

def print_message(message='', level=0):
    log_message.append(message)
    if args.verbose >= level:
        print(message)

def object_check(x):
    if os.path.isdir(x):
        return False
    with open(x, 'r') as f:
        lines = f.readlines()
        if (len(lines) == 0):
            return False
    return True

def read_content(file_content):
    lines = file_content
    labels, x0s, y0s, x1s, y1s = [], [], [], [], []
    for line in lines:
        line = line.strip()
        elements = line.split(' ')
        label, x0, y0, x1, y1 = elements
        x0 = int(x0)
        y0 = int(y0)
        x1 = int(x1)
        y1 = int(y1)
        labels.append(label)
        x0s.append(x0)
        y0s.append(y0)
        x1s.append(x1)
        y1s.append(y1)
    return (labels, x0s, y0s, x1s, y1s)

def read_voc(img_id):
    path = d[d['id'] == img_id]['coco'].tolist()[0]
    with open(path, 'r') as f:
        content = f.readlines()
        return read_content(content)

    
class AppDataset(Dataset):
    
    def __init__(self, d, transforms=None):
        super().__init__()
        
        self.image_ids = d["id"].unique()
        self.df = d
        self.transforms = transforms
        
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        labels, x0s, y0s, x1s, y1s = read_voc(image_id)
        
        image = cv2.imread(self.df[self.df['id'] == image_id]['image'].tolist()[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image/255.0
        image = image.transpose(2,0,1)
        image = torch.from_numpy(image).float()
       
        boxes = np.array(list(zip(x0s, y0s, x1s, y1s)))
        if boxes.shape == (0,):
            raise Exception(f'Wrong format {image_id}')
        labels = torch.tensor([label2id[l] for l in labels], dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        num_objs = len(boxes)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target['boxes'] = torch.from_numpy(boxes)
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target["area"] = torch.from_numpy(area)
        target["iscrowd"] = iscrowd
        return image, target
    
    def __len__(self):
        return self.image_ids.shape[0]

    
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, p=0.25),
        A.LongestMaxSize(max_size=800, p=1.0),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_test_transform():
    return A.Compose([
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def collate_fn(batch):
    return tuple(zip(*batch))


def mAP(test_dataset, test_data_loader, threshold=0.3):
    it_data_loader = iter(test_data_loader)
    det_image_ids = list()
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()
    idx = 0

    for i in range(len(test_dataset)):
        images, targets = next(it_data_loader)
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        image_ids = [k['image_id'] for k in targets]

        preds = model(images)
        preds = [{k: v.to(cpu) for k, v in t.items()} for t in preds]

        boxes = targets[idx]['boxes'].cpu().numpy().astype(np.int32)
        image = images[idx].cpu().numpy()
        image = image.transpose(1,2,0)

        pred_scores, pred_boxes, pred_classes = get_prediction(preds[idx], threshold)
        pred_boxes = pred_boxes.astype(np.int32)

        det_boxes_batch = torch.from_numpy(pred_boxes)
        det_labels_batch = torch.from_numpy(pred_classes)
        det_scores_batch = torch.from_numpy(pred_scores)

        boxes = targets[idx]['boxes']
        labels = targets[idx]['labels']

        det_boxes.append(det_boxes_batch)
        det_labels.append(det_labels_batch)
        det_scores.append(det_scores_batch)

        true_boxes.append(boxes)
        true_labels.append(labels)
        det_image_ids.extend(image_ids)
    copy2voc(test_dataset, det_image_ids, true_labels, true_boxes, det_labels, det_scores, det_boxes)
    subprocess.run(["python", "pascalvoc.py", "-t", "0.3"], cwd="./mAP")
        
def copy2voc(selected_dataset, det_image_ids, true_labels, true_boxes, det_labels, det_scores, det_boxes):
    shutil.rmtree('./mAP/detections')
    shutil.rmtree('./mAP/groundtruths')
    shutil.rmtree('./mAP/results')
    os.makedirs('./mAP/detections')
    os.makedirs('./mAP/groundtruths')
    os.makedirs('./mAP/results')

    for i, tl, tb, dl, ds, db in zip(det_image_ids, true_labels, true_boxes, det_labels, det_scores, det_boxes):
        idx = i.cpu().numpy()[0]
        fn = selected_dataset.image_ids[idx] + '.txt'
        # Write groundtruth content
        gt_fn = os.path.join('./mAP/groundtruths', fn)
        with open(gt_fn, 'w') as gt_file:
            lines = []
            for j in range(len(tl)):
                label = id2label[tl[j].cpu().numpy().tolist()]
                x0 = tb[j].cpu().numpy()[0]
                y0 = tb[j].cpu().numpy()[1]
                x1 = tb[j].cpu().numpy()[2]
                y1 = tb[j].cpu().numpy()[3]
                line = f'{label} {x0} {y0} {x1} {y1}'
                lines.append(line)
            content = '\n'.join(lines)
            gt_file.writelines(content)

        # Write detection content
        dt_fn = os.path.join('./mAP/detections', fn)
        with open(dt_fn, 'w') as dt_file:
            lines = []
            for j in range(len(dl)):
                label = id2label[dl[j].cpu().numpy().tolist()]
                score = ds[j].cpu().numpy()
                x0 = db[j].cpu().numpy()[0]
                y0 = db[j].cpu().numpy()[1]
                x1 = db[j].cpu().numpy()[2]
                y1 = db[j].cpu().numpy()[3]
                line = f'{label} {score} {x0} {y0} {x1} {y1}'
                lines.append(line)
            content = '\n'.join(lines)
            dt_file.writelines(content)
            
            
def get_prediction(pred_dict, threshold):
    pred_class = pred_dict['labels'].detach().numpy()
    pred_boxes = pred_dict['boxes'].detach().numpy()
    pred_score = pred_dict['scores'].detach().numpy()
    pred_t = pred_score >= threshold
    pred_boxes = pred_boxes[pred_t]
    pred_class = pred_class[pred_t]
    pred_scores = pred_score[pred_t]
    return pred_scores, pred_boxes, pred_class


"""MAIN"""

parser = argparse.ArgumentParser(description='Configure for training pipeline for ACNE detection')

parser.add_argument('--data', '-d', default='./bbox', type=str, help='Input path to VOC files')
parser.add_argument('--epochs', '-e', default=100, type=int, help='Number of epochs to train - 100')
parser.add_argument('--batch_size','-b', default=1, type=int, help='Number of batch_size - 1')
parser.add_argument('--optimizer' , '-o', default='adam', type=str, help='Use specific optimizer function - adam')
parser.add_argument('--learning_rate' , '-lr', default=0.001, type=float, help='Use specific learning rate - 0.001')
# parser.add_argument('--image_size', '-s' , default=224, type=int, help='Use to define input image size - (224,224)')
parser.add_argument('--pretrained', '-pt' , default=True, type=bool, help='Use pretrained weight - True')
parser.add_argument('--cuda' , '-c', default=0, type=int, help='Use specific CuDA - 0')
parser.add_argument('--num_workers' , '-w', default=8, type=int, help='Use number of workers - 8')
parser.add_argument('--verbose', '-v', default=10, type=int, help='Log the process and information - 0')
parser.add_argument('--continue_training', default=True, type=bool, help='Continue training model from the last execution - True')

parser.add_argument('--save_epoch', default=100, type=int, help='Number of epoch to save model - 100')
parser.add_argument('--model_store', default='./models', type=str, help='Place to store model - ./models')
parser.add_argument('--print_freq', default=100, type=int, help='Number of steps to printout - 100')

parser.add_argument('--debug', default=False, type=bool, help='Debug Mode - False')
parser.add_argument('--debug_size', default=100, type=int, help='Debug Size images - 100')

# Run
args = parser.parse_args(sys.argv[1:])

print_message("PyTorch Version: %s" % torch.__version__)
print_message("Torchvision Version: %s" % torchvision.__version__)

print_message('\nArgument Parser')
for k, v in vars(args).items():
    print_message('{:20}:    {:}'.format(k, v), 2)
    
# CONSTANT
id2label = {
    0: 'vet_tham',
    1: 'thuong_viem',
    2: 'seo_mun',
    3: 'coi',
    4: 'thuong_nang'
}

label2id = {
    'dát_tăng_sắc_tố_(vết_thâm)': 0,
    'sang_thương_viêm_(sẩn,_mụn_mủ,_mảng_viêm_đỏ)': 1,
    'sẹo_mụn_(lõm,_lồi)': 2,
    'còi_(đóng/mở)': 3,
    'sang_thương_nang_và_nốt': 4,
}

id2color = {
    0: (255, 0, 0), #red
    1: (255, 128, 0), #Orange
    2: (255, 153, 255), #pink
    3: (0, 255, 0), #green
    4: (0, 0, 255) #blue
}

print_message('=======================================', 1)
print_message('===============PREPARATION=============', 1)
print_message('=======================================', 1)

# MAIN PIPELINE
img_ids = os.listdir(args.data)
d = pd.DataFrame({'id': img_ids})
d['coco'] = d['id'].apply(lambda x: os.path.join(args.data, x))
d['coco_check'] = d['coco'].apply(lambda x: os.path.exists(x))
d['image'] = d['id'].apply(lambda x: os.path.join('./image', x + '.jpg'))
d['image_check'] = d['image'].apply(lambda x: os.path.exists(x))
d['object_check'] = d['coco'].apply(lambda x: object_check(x))
d = d[(d['coco_check'] == True) & (d['image_check'] == True) & (d['object_check'] == True)]
d = d.reset_index(drop=True)

if args.debug:
    d = d.sample(frac=1).reset_index(drop=True)
    d = d[:args.debug_size]

print_message(f'Number of qualified images: {len(d)}', 1)

# Split train/test/val
train_df, test_df = train_test_split(d, test_size=0.3)
print_message(f'Number of train/val/test {len(train_df)}/{len(test_df)}', 1)


# Apply transformation
train_dataset = AppDataset(train_df, get_train_transform())
test_dataset = AppDataset(test_df, get_test_transform())

# Dataloader
train_data_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
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

# GPU selection
cpu = torch.device('cpu')
device = torch.device("cuda:%s" % args.cuda if torch.cuda.is_available() else "cpu")
print_message(f'Selected GPU: {device}')


# Num classes
NUM_CLASSES = len(label2id)
print_message(f'Number of classes: {NUM_CLASSES}')

print_message('\n===============TRAINING=============\n', 1)

start_epoch = 1

cache_models = os.listdir('./models')
cache_index = [int(re.findall(r'_epoch(\d+)', i)[0]) for i in cache_models]

if (len(cache_index) > 0) and (args.continue_training is True):
    print_message('Continuous training...', 1)
    max_idx = max(cache_index)
    print_message(f'Found {len(cache_index)} models cached, maximum index model: {max_idx}', 1)
    model = torch.load(f'./models/model_epoch{max_idx}.pkl', map_location='cuda:%s' % args.cuda)
    model = model.train()
    start_epoch = max_idx + 1
else:
    # 1. Pretrained
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
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
optimizer = torch.optim.Adam(params, lr=args.learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

for epoch in range(start_epoch, args.epochs + 1):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=args.print_freq)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, test_data_loader, device=device)
    mAP(test_dataset, test_data_loader)
    if epoch % args.save_epoch == 0:
        torch.save(model, f'./models/model_epoch{epoch}.pkl')
        
print_message('Pipeline finished', 1)
log_message_str = '\n'.join(log_message)
