"""Example of pykitti.raw usage."""
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pykitti
from run_model import load_model,run_model, plot_result
import cv2

def view_lidar_basic(points):
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

def pointcloud2image(pointcloud, imheight, imwidth, Tr, K):

    pointcloud = pointcloud[pointcloud[:, 0] > 0] #remove points behind the camera
    pointcloud = np.hstack([pointcloud[:, :3], np.ones(pointcloud.shape[0]).reshape((-1,1))]) #put ones instead reflection on the 4th colomn
    cam_xyz = Tr.dot(pointcloud.T) #transform velo pointcloud to camera 2 coordinate system
    cam_xyz = cam_xyz[:, cam_xyz[2] > 0] # remove some points that are behind the camera after doing the projection
    depth = cam_xyz[2].copy() #use it later for ploting
    # divide by Zc
    cam_xyz[0] /= cam_xyz[2]
    cam_xyz[1] /= cam_xyz[2]
    cam_xyz[2] /= cam_xyz[2]
    cam_xyz = np.delete(cam_xyz,3,0)
    #now we have [Xc/Zc Yc/Zc 1]
   
    
    projection = K.dot(cam_xyz) #find the u,v of the image 
    pixel_coordinates = np.round(projection.T, 0)[:, :2].astype('int') # make it integer 
    indices = np.where((pixel_coordinates[:, 0] < imwidth)
                       & (pixel_coordinates[:, 0] >= 0)
                       & (pixel_coordinates[:, 1] < imheight)
                       & (pixel_coordinates[:, 1] >= 0)
                      )
    pixel_coordinates = pixel_coordinates[indices] #take out points outside the image
    depth = depth[indices]
    return pixel_coordinates, depth

def plot_pointcloud2image(cam,pixel_coordinates,depth):
    (IMG_W,IMG_H) = cam.size
    plt.axis([0,IMG_W,IMG_H,0])
    plt.imshow(cam)
    plt.scatter([pixel_coordinates[:,0]],[pixel_coordinates[:,1]],c=[depth],cmap='rainbow_r',alpha=0.5,s=2)
    plt.title('Projection')
    plt.show()

def featureDetection():
    thresh = dict(threshold=25, nonmaxSuppression=True);
    fast = cv2.FastFeatureDetector_create(**thresh)
    return fast
    
def featureTracking(img_1, img_2, p1):

    lk_params = dict( winSize  = (21,21),
                      maxLevel = 3,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    p2, st, err = cv2.calcOpticalFlowPyrLK(img_1, img_2, p1, None, **lk_params)
    st = st.reshape(st.shape[0])
    ##find good one
    p1 = p1[st==1]
    p2 = p2[st==1]
    return p1,p2

model_config = '/Users/robertkrutsch/Documents/Code/VD/vehicle-detection/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml'
model_path = '/Users/robertkrutsch/Documents/Code/VD/best.pth'

basedir = '/Users/robertkrutsch/Documents/Code/VD/data/kiti_odo/'
date = '2011_09_26'
drive = '0001'
nr_frames = 10

dataset = pykitti.raw(basedir, date, drive, frames=range(0, nr_frames, 1))
model = load_model(model_config,model_path)


#GET FIRST IMAGE
cam2_prev = dataset.get_cam2(0)
cam2_prev_cv2 = cv2.cvtColor(np.array(cam2_prev), cv2.COLOR_RGB2BGR)#transform the thing to opencv format
scores1_prev, boxes1_prev, labels1_prev = run_model(cam2_prev, model) #find vehicles and so on
gray_prev = cv2.cvtColor(cam2_prev_cv2, cv2.COLOR_BGR2GRAY)#features work on grayscale

#GET SECOND IMAGE
cam2_cur = dataset.get_cam2(i)    
cam2_cur_cv2 = cv2.cvtColor(np.array(cam2_cur), cv2.COLOR_RGB2BGR)#transform the thing to opencv format    
scores1_cur, boxes1_cur, labels1_cur = run_model(cam2_cur, model)
gray_cur = cv2.cvtColor(cam2_cur_cv2, cv2.COLOR_BGR2GRAY)#features work on grayscale


#RUN OPTICAL FLOW 
detector = featureDetection()
kp1      = detector.detect(cam2_cur_cv2)
p1       = np.array([ele.pt for ele in kp1],dtype='float32')
p1, p2   = featureTracking(gray_prev, gray_cur, p1)





for i in range(1,nr_frames):

    cam2_cur = dataset.get_cam2(i)    
    cam2_cur_cv2 = cv2.cvtColor(np.array(cam2_cur), cv2.COLOR_RGB2BGR)#transform the thing to opencv format    
    scores1_cur, boxes1_cur, labels1_cur = run_model(cam2_cur, model)
    gray_cur = cv2.cvtColor(cam2_cur_cv2, cv2.COLOR_BGR2GRAY)#features work on grayscale


    detector = featureDetection()
    kp1      = detector.detect(cam2_cur_cv2)
    p1       = np.array([ele.pt for ele in kp1],dtype='float32')
    p1, p2   = featureTracking(gray_prev, gray_cur, p1)


    cam2_prev_cv2 = cam2_cur_cv2
    scores1_prev  =  scores1_cur
    boxes1_prev   =  boxes1_cur
    labels1_prev  = labels1_cur
    gray_prev = gray_cur



    #velo = dataset.get_velo(i)
    #pixel_coordinates,depth = pointcloud2image(velo, cam2_cur.height, cacam2_curm2.width, dataset.calib.T_cam2_velo, dataset.calib.K_cam2)
    #plot_pointcloud2image(cam2_cur,pixel_coordinates,depth)
    #plot_result(cam2_cur,boxes1,labels1, scores1,'/Users/robertkrutsch/Documents/Code/VD/data/kiti/labeled_images/'+str(i)+'.png') #plot detection results
    #plot_pointcloud2image(cam2_cur,p2,np.ones(p2.shape[0])) # plot the points tracked

print("Done")


