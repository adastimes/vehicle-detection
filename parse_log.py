'''
quick plot of the losses in the log 
'''
import json
import matplotlib.pyplot as plt
import re

train_loss = []
train_loss_vfl = []
train_loss_bbox = []
train_loss_giou = []

with open('/home/rob/Work/vd/output/rtdetrv2_r18vd_120e_coco/log.txt','r') as f:
    for line in f:
        s = line.rstrip()
        d = json.loads(s)
        train_loss.append(d['train_loss'])
        train_loss_vfl.append(d['train_loss_vfl'])
        train_loss_bbox.append(d['train_loss_bbox'])
        train_loss_giou.append(d['train_loss_giou'])

fig = plt.figure()

plt.plot(train_loss)

plt.show()

print("Done")
