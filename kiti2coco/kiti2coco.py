import functools
import json
import os
import random
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
import pycocotools
from pycocotools import mask
import numpy as np
import pandas as pd
import uuid
import cv2
from pprint import pprint
from PIL import Image

import itertools

from PIL import Image

TRAIN_ROOT = '/Users/robertkrutsch/Documents/Code/VD/data/kiti/training'
LABEL_PATH = '/Users/robertkrutsch/Documents/Code/VD/data/kiti/label_2'
ANNO_PATH = '/Users/robertkrutsch/Documents/Code/VD/data/kiti/annotations'


'''
# in case you want to reuse the coco clases as they are and just want to map kitti towards the coco classes
# first are the kitti classes , the number is the id of the coco classes
Cathegory_Match = {
'Car':3,#
'Van':3,#
'Truck':8,
'Pedestrian':1,#
'Person_sitting':1,#
'Cyclist':2,
'Tram':7
}
'''

# this is a from scratch mapping 1- person, 2- car, 3 - bycicle, 4- Train/Tram , 5 - Truck
# categories are from scratchs and close to the ones from coco but not exactly
Cathegory_Match = {
'Car':2,#
'Van':2,#
'Truck':5,
'Pedestrian':1,#
'Person_sitting':1,#
'Cyclist':3,
'Tram':4
}


def plot_results_kiti(img_path, xmin, ymin, width, height,obj_type):

    pil_img = Image.open(img_path)

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    ax.add_patch(plt.Rectangle((xmin, ymin), width, height ,
                                   fill=False, color=[0.000, 0.447, 0.741], linewidth=3))

    ax.text(xmin, ymin, obj_type, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def prep_annotation_json(images_list, json_file_list):
    images = []
    annotations = []

    for idx, img_name in enumerate(sorted(images_list)):
            #if idx==10:
            #    break
            img_path = os.path.join(TRAIN_ROOT, img_name)
            height, width = cv2.imread(img_path).shape[:2]       
            img_name_wo_suffix, file_extension = os.path.splitext(img_name)
            img_id = int(img_name_wo_suffix) 
            label_file_path = os.path.join(LABEL_PATH, '{}.txt'.format(img_name_wo_suffix))
            labels = pd.read_csv(label_file_path, delimiter=' ', header=None).values

            #get annotations from file    
            annotations_file = []
            for idx, label in enumerate(labels):
                    
                    if (label[0]!='DontCare') and (label[0]!='Misc'):
                        category_id = Cathegory_Match[label[0]] #transform category
                        top_x, top_y, bottom_x, bottom_y = label[4:8] #get bounding box
                        area = (bottom_x - top_x) * (bottom_y - top_y) # get area

                        xmin = int(top_x)
                        ymin = int(bottom_y)
                        xmax = int(bottom_x)
                        ymax = int(top_y)

                        #plot_results_kiti(img_path,xmin, ymin, xmax-xmin,ymax-ymin, label[0])

                        annotations_file.append({ # populate annotation dict like in coco
                            'segmentation': None,
                            'area': area,
                            'iscrowd': 0,
                            'image_id': img_id,
                            'bbox': [top_x, top_y, bottom_x - top_x, bottom_y - top_y],
                            'category_id': category_id,
                            'id': '{}_{}'.format(img_id, idx), # in coco this are numbers, might need to redo this
                        })

            images.append({
                'license': 1,
                'file_name': img_name.replace('/', '_'),
                'coco_url': '',
                'height': height,
                'width': width,
                'date_captured': '',
                'flickr_url': '',
                'id': img_id
            })
            annotations.extend(annotations_file)


    dataset = {
            'info': {
                'description': 'KITTI 2 Coco',
                'url': 'john.adas.doe@gmail.com',
                'version': '0.1',
                'year': 2025,
                'contributor': 'BigDog',
                'date_created': '2025/1/1'
            },
            'licenses': [{
                'url':'john.adas.doe',
                'id':1,
                'name':'Fortza Pandurii!'
            }],
            'images': images,
            'annotations': annotations,
        }

    #copy categories in case you want to keep the kitti to coco mapping
    #data = json.load(open('/Users/robertkrutsch/Documents/Code/VD/data/coco2017/annotations/instances_train2017.json','r'))
    #dataset['categories'] = data['categories']

    dataset['categories'] = [{'supercategory': 'person', 'id': 1, 'name': 'person'},
            {'supercategory': 'vehicle', 'id': 2, 'name': 'car'},
            {'supercategory': 'vehicle', 'id': 3, 'name': 'bicycle'},
            {'supercategory': 'vehicle', 'id': 4, 'name': 'train'},
            {'supercategory': 'vehicle', 'id': 5, 'name': 'truck'}]


    json.dump(dataset, open(json_file_list, 'w'), indent=0)

img_list = os.listdir(TRAIN_ROOT)
random.shuffle(img_list)
split = int(len(img_list) * 0.9)
train_file_list = img_list[:split]
val_list_list = img_list[split:]

prep_annotation_json(train_file_list, os.path.join(ANNO_PATH, 'instances_train.json'))
prep_annotation_json(val_list_list, os.path.join(ANNO_PATH,'instances_val.json'))

print('Done')

'''
kitti - coco

Car - 2
Cyclist -1
Misc
Pedestrian - 0
Person_sitting - 0, pus original din person
Tram - 6 , original e train 
Truck - 7
Van - 2 , original e car 

CLASSES_RT = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',  'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
'''