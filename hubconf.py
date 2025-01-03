"""Copyright(c) 2024 lyuwenyu. All Rights Reserved.
"""


import os
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).absolute().parent / 'rtdetrv2_pytorch'
sys.path.append(str(ROOT))

from src.core import YAMLConfig

import torch
import torch.nn as nn


from PIL import Image
import requests
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt

from ptflops import get_model_complexity_info
from torchvision.models import vit_b_16
from torchvision.models import resnet50

dependencies = ['torch', 'torchvision',]
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
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

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def _load_checkpoint(path: str, map_location='cpu'):
    scheme = urlparse(str(path)).scheme
    if not scheme:
        state = torch.load(path, map_location=map_location)
    else:
        state = torch.hub.load_state_dict_from_url(path, map_location=map_location)
    return state


def _build_model(args, ):
    """main
    """
    cfg = YAMLConfig(args.config)

    if args.resume:
        checkpoint = _load_checkpoint(args.resume, map_location='cpu') 
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


CONFIG = {
    # rtdetr
    'rtdetr_r18vd': {
        'config': ROOT / 'configs/rtdetr/rtdetr_r18vd_6x_coco.yml',
        'resume': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth',
    },
    'rtdetr_r34vd': {
        'config': ROOT / 'configs/rtdetr/rtdetr_r34vd_6x_coco.yml',
        'resume': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth',
    },
    'rtdetr_r50vd_m': {
        'config': ROOT / 'configs/rtdetr/rtdetr_r50vd_m_6x_coco.yml',
        'resume': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth',
    },
    'rtdetr_r50vd': {
        'config': ROOT / 'configs/rtdetr/rtdetr_r50vd_6x_coco.yml',
        'resume': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth',
    },
    'rtdetr_r101vd': {
        'config': ROOT / 'configs/rtdetr/rtdetr_r101vd_6x_coco.yml',
        'resume': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth',
    },

    # rtdetrv2
    'rtdetrv2_r18vd': {
        'config': ROOT / 'configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml',
        'resume': 'https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth',
    },
    'rtdetrv2_r34vd': {
        'config': ROOT / 'configs/rtdetrv2/rtdetrv2_r34vd_120e_coco.yml',
        'resume': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r34vd_120e_coco_ema.pth',
    },
    'rtdetrv2_r50vd_m': {
        'config': ROOT / 'configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml',
        'resume': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_7x_coco_ema.pth',
    },
    'rtdetrv2_r50vd': {
        'config': ROOT / 'configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml',
        'resume': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth',
    },
    'rtdetrv2_r101vd': {
        'config': ROOT / 'configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml',
        'resume': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth',
    },
}


# rtdetr
def rtdetr_r18vd(pretrained=True,):
    args = type('Args', (), CONFIG['rtdetr_r18vd'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )


def rtdetr_r34vd(pretrained=True,):
    args = type('Args', (), CONFIG['rtdetr_r34vd'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )


def rtdetr_r50vd_m(pretrained=True):
    args = type('Args', (), CONFIG['rtdetr_r50vd_m'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )


def rtdetr_r50vd(pretrained=True):
    args = type('Args', (), CONFIG['rtdetr_r50vd'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )


def rtdetr_r101vd(pretrained=True):
    args = type('Args', (), CONFIG['rtdetr_r101vd'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )


# rtdetrv2
def rtdetrv2_r18vd(pretrained=True,):
    args = type('Args', (), CONFIG['rtdetrv2_r18vd'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )


def rtdetrv2_r34vd(pretrained=True,):
    args = type('Args', (), CONFIG['rtdetrv2_r34vd'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )


def rtdetrv2_r50vd_m(pretrained=True):
    args = type('Args', (), CONFIG['rtdetrv2_r50vd_m'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )


def rtdetrv2_r50vd(pretrained=True):
    args = type('Args', (), CONFIG['rtdetrv2_r50vd'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )


def rtdetrv2_r101vd(pretrained=True):
    args = type('Args', (), CONFIG['rtdetrv2_r101vd'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )

trans = T.Compose([
    #T.Resize([640,640]),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class DETRResnet50(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """

    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h),
                'pred_boxes': self.linear_bbox(h).sigmoid()}
    

def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[
        -1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    return probas[keep], bboxes_scaled

def detect_rt(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[
        -1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img, torch.tensor(im.size))
    # 0 - labels, 1 - bb, 2 - scores ( there are 300 of them)
    keep = outputs[2]>0.7

    return outputs[2][keep], outputs[1][keep], outputs[0][keep]

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def plot_results_rt(pil_img, boxes,labels,scores):
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
    plt.show()

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

#rtdetrv2_s = rtdetrv2_r18vd(True)
#rtdetrv2_m_r34 = rtdetrv2_r34vd
#rtdetrv2_m_r50 = rtdetrv2_r50vd_m
rtdetrv2_s = rtdetrv2_r50vd(True)
#rtdetrv2_x = rtdetrv2_r101vd

detr = DETRResnet50(num_classes=91)
state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True)
detr.load_state_dict(state_dict)
detr.eval()

url = '/Users/robertkrutsch/Documents/Code/VD/data/kiti/training/000000.png'
#url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
#im = Image.open(requests.get(url, stream=True).raw)
im = Image.open(url)


#scores, boxes = detect(im, detr, trans)

scores1, boxes1, labels1 = detect_rt(im, rtdetrv2_s, trans)

#plot_results(im, scores, boxes)

plot_results_rt(im,boxes1,labels1, scores1)

print("\n -> DONE")



