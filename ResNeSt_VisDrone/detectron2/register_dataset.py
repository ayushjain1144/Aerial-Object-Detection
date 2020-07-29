import os
import numpy as np
import json
import cv2
import random
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog,MetadataCatalog

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

base_dir = '/home/user1/aerial_detection/Aerial-Object-Detection/aiskyeye_dataset'
train_dir = os.path.join(base_dir, 'VisDrone2019-DET-train')
val_dir = os.path.join(base_dir, 'VisDrone2019-DET-val')
val_test_dev_dir = os.path.join(base_dir, 'Visdrone2019-DET-test-dev')

train_annotations = os.path.join(train_dir, 'annotations')
val_annotations = os.path.join(val_dir, 'annotations')
val_images_dir = os.path.join(val_dir, "images")
val_test_dev_annotations = os.path.join(val_test_dev_dir, 'annotations')

train_annotations_list = sorted(os.listdir(train_annotations))
val_annotations_list = sorted(os.listdir(val_annotations))
#val_test_dev_annotations_list = sorted(os.listdir(val_test_dev_annotations))

train_images_dir = os.path.join(train_dir, 'images')
train_images_list = sorted(os.listdir(train_images_dir))

def get_dicts(set, base_dir=base_dir):  # set=train/val/test-dev
    dataset_dicts = []
    img_dir = os.path.join(base_dir, 'VisDrone2019-DET-' + set, 'images')
    ann_dir = os.path.join(base_dir, 'VisDrone2019-DET-' + set, 'annotations')
    for fn in sorted(os.listdir(img_dir)):
        record = {}
        filename = os.path.join(img_dir, fn)
        idx = fn[:-4]
        height,width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        ann = open(os.path.join(ann_dir, fn[:-3] + 'txt'), 'r')
        
        objs = []
        for line in ann.readlines():
            if(line[-2]==','):
                line = line[:-2]
            try:
                x1, y1, w, h,_,o,_,_ = line.split(sep=',')
                
                if w == '0'  or h == '0' or int(o)<=0 or int(o)>=11:
                    continue
            except:
                print(line)

            obj = {
                "bbox": [int(x1), int(y1), int(w), int(h)],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": int(o) -1,
            }
            
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def register_datasets():
    for d in ["train", "val", "test-dev"]:
        DatasetCatalog.register("visdrone_" + d, lambda d=d: get_dicts(d))
        MetadataCatalog.get("visdrone_" + d).set(thing_classes=['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'])
    visdrone_metadata = MetadataCatalog.get("visdrone_train")
