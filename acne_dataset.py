from torch.utils.data import Dataset
import os
import json
import cv2
import torch
import numpy as np


class AcneDataset(Dataset):
    
    def __init__(self, dataset_loc, transforms=None):
        super().__init__()
        self.dataset_loc = dataset_loc
        self.image_folder = os.path.join(dataset_loc, 'image')
        self.coco_json = json.load(open(os.path.join(dataset_loc, 'coco_instances.json'),'r'))
        self.category = self.coco_json['categories']
        self.images = self.coco_json['images']
        self.annotations = self.coco_json['annotations']
        self.transforms = transforms
        self.reload_annotation()
        
    def reload_annotation(self):
        self.imageid_annotations = {}
        for anno in self.annotations:
            imgid = anno['image_id']
            self.imageid_annotations[imgid] = {
                'classes': [],
                'boxes': []
            }
        for anno in self.annotations:
            imgid = anno['image_id']
            cate = anno['category_id']
            bbox = anno['bbox']
            self.imageid_annotations[imgid]['classes'].append(cate)
            self.imageid_annotations[imgid]['boxes'].append(bbox)
         
        keys = range(len(self.imageid_annotations))
        values = list(self.imageid_annotations.keys())
        self.correct_idx = dict(zip(keys,values))
        # print(self.correct_idx)
        
    def __getitem__(self, idx):
        index = self.correct_idx[idx]
        image_id = self.images[index]['id']
        image_fn = self.images[index]['file_name']
        width, height = self.images[index]['width'], self.images[index]['height']
        
        classes = self.imageid_annotations[image_id]['classes']
        boxes = self.imageid_annotations[image_id]['boxes']
        voc_boxes = []
        for b in boxes:
            x,y,w,h = b
            x0,y0,x1,y1 = x,y, x+w, y+h
            voc_boxes.append([x0,y0,x1,y1])
        
        image = cv2.imread(os.path.join(self.image_folder, image_fn))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image/255.0
        image = image.transpose(2,0,1)
        image = torch.from_numpy(image).float()
       
        boxes = np.array(voc_boxes)
        if boxes.shape == (0,):
            raise Exception(f'Wrong format {image_id}')
        labels = torch.tensor(classes, dtype=torch.int64)
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
        return len(self.imageid_annotations)