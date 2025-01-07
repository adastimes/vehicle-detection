"""Example of pykitti.raw usage."""
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pykitti
from run_model import load_model,run_model, plot_result

def view_lidar(points):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.set_box_aspect((np.ptp(points[:,0]),np.ptp(points[:,1]),np.ptp(points[:,2])))
    ax.scatter(points[:,0],points[:,1],points[:,2],s=0.01)
    ax.grid(False)
    ax.axis('off')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=40, azim=180)

def pointcloud2image(points,im)


model_config = '/Users/robertkrutsch/Documents/Code/VD/vehicle-detection/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml'
model_path = '/Users/robertkrutsch/Documents/Code/VD/best.pth'

basedir = '/Users/robertkrutsch/Documents/Code/VD/data/kiti_odo/'
date = '2011_09_26'
drive = '0001'
nr_frames = 10

dataset = pykitti.raw(basedir, date, drive, frames=range(0, nr_frames, 1))
model = load_model(model_config,model_path)

for i in range(1):
    cam2 = dataset.get_cam2(i)
    velo = dataset.get_velo(i)
    scores1, boxes1, labels1 = run_model(cam2, model)
    #plot_result(cam2,boxes1,labels1, scores1,'/Users/robertkrutsch/Documents/Code/VD/data/kiti/labeled_images/'+str(i)+'.png')

print("Done")

