import os
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = '/Users/robertkrutsch/Documents/Code/VD/vehicle-detection/rtdetrv2_pytorch'
sys.path.append(str(ROOT))

from src.core import YAMLConfig

import torch
import torch.nn as nn


from PIL import Image
import requests
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

CLASSES_RT = [
'Pedestrian','Car','Cyclist','Tram','Truck'
]

def load_model(config,model_path):
    
    cfg = YAMLConfig(config)

    checkpoint = torch.load(model_path, map_location='cpu')

    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']

        # NOTE load train mode state
    cfg.model.load_state_dict(state)


    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    return Model()

def run_model(im, model):

    trans = T.Compose([
    #T.Resize([640,640]),
    T.ToTensor()
    ])

    img = trans(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img, torch.tensor(im.size))
    # 0 - labels, 1 - bb, 2 - scores ( there are 300 of them)
    keep = outputs[2]>0.7

    return outputs[2][keep], outputs[1][keep], outputs[0][keep]

def plot_result(pil_img, boxes,labels,scores, output_file):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for l,s,(xmin, ymin, xmax, ymax), c in zip(labels,scores,boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{CLASSES_RT[l]}: {s:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
        
    plt.axis('off')
    plt.savefig(output_file)
    plt.close()
    #plt.show()

'''
INP_ROOT = '/Users/robertkrutsch/Documents/Code/VD/data/kiti/testing/'
OUT_ROOT = '/Users/robertkrutsch/Documents/Code/VD/data/kiti/labeled_images/'
config = '/Users/robertkrutsch/Documents/Code/VD/vehicle-detection/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml'
model_path = '/Users/robertkrutsch/Documents/Code/VD/best.pth'


img_list = sorted(os.listdir(INP_ROOT))
rtdetrv2_s = load_model(config,model_path) # init model


for f,idx in zip(img_list,range(img_list.__len__())):
    im = Image.open(INP_ROOT + f)
    scores1, boxes1, labels1 = run_model(im, rtdetrv2_s)
    plot_results_rt(im,boxes1,labels1, scores1,OUT_ROOT + f)

print("Done")
'''