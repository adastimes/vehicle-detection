import json
from collections import defaultdict
import matplotlib.pyplot as plt
import os 
import shutil



def copy_kitti_images_to_coco(dir_kiti = '/Users/robertkrutsch/Documents/Code/VD/data/kiti/2011_09_26/2011_09_26_drive_0001_sync/image_02/data', 
                              dir_coco='/Users/robertkrutsch/Documents/Code/VD/data/kiti/2011_09_26/2011_09_26_drive_0001_sync/image_02/data_coco', 
                              drive_id=1):
    '''
        coco expects 12 digit number!
        output format is : 0000 ddd fffff ( 4 leading zeros , drive as 3 digits, 5 frame number)

    '''
    if os.path.exists(dir_coco):
        return 0

    try: # make the output directory that mimics coco
        os.mkdir(dir_coco)
        print(f"Directory '{dir_coco}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dir_coco}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{dir_coco}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


    for (dirpath, dirnames, filenames) in os.walk(dir_kiti):        
        for f in filenames:

            filename, file_extension = os.path.splitext(f)
            filenname_coco = '0000' + str(drive_id).zfill(3) + filename.strip('0').zfill(5) + file_extension #12 digits long

            assert(filenname_coco.__len__()==16)

            shutil.copy(dir_kiti + '/' + f, dir_coco + '/' + filenname_coco)

    return 1


ret_coode = copy_kitti_images_to_coco()
print(ret_coode)



'''
d = json.load(open('/Users/robertkrutsch/Documents/Code/VD/data/coco2017/annotations/instances_train2017_small.json','r'))
a = []
for ann in d['annotations']:
    a.append(ann['category_id'])


#### PLOT the loss for the epocs in LOG!

train_loss = []
LOG = []
with open('/Users/robertkrutsch/Documents/Code/VD/rtdetr/RT-DETR/rtdetr_pytorch/output/rtdetr_r50vd_6x_coco/log.txt','r') as sample:
    for line in sample:
        line = json.loads(line.strip())
        train_loss.append(line['train_loss'])
        LOG.append(line)

plt.plot(train_loss)
'''

##### Make a shorter val and train datasets files

data = json.load(open('/Users/robertkrutsch/Documents/Code/VD/data/coco2017/annotations/instances_train2017.json','r'))
#dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])

imgToAnns = defaultdict(list)
anns = {}
imgs = {}

for ann in data['annotations']:
    imgToAnns[ann['image_id']].append(ann)
    anns[ann['id']] = ann

for img in data['images']:
    imgs[img['id']] = img


imgs_tmp = []
anns_tmp = []
for i in range(100):
    imgs_tmp.append(data['images'][i])
    for elem in imgToAnns[data['images'][i]['id']]:
        if(elem['category_id']<70):
            anns_tmp.append(elem)


del data['categories'][61:]

data_out = {}
data_out['info'] = data['info']
data_out['licenses'] = data['licenses']
data_out['categories'] = data['categories']
data_out['images'] = imgs_tmp
data_out['annotations'] = anns_tmp

with open('/Users/robertkrutsch/Documents/Code/VD/data/coco2017/annotations/instances_train2017_small.json', 'w') as fp:
    json.dump(data_out, fp)

print("Done")

######################VALIDATION#######################

data = json.load(open('/Users/robertkrutsch/Documents/Code/VD/data/coco2017/annotations/instances_val2017.json','r'))
#dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])

imgToAnns = defaultdict(list)
anns = {}
imgs = {}

for ann in data['annotations']:
    imgToAnns[ann['image_id']].append(ann)
    anns[ann['id']] = ann

for img in data['images']:
    imgs[img['id']] = img


imgs_tmp = []
anns_tmp = []
for i in range(100):
    imgs_tmp.append(data['images'][i])
    for elem in imgToAnns[data['images'][i]['id']]:
        if(elem['category_id']<70):
            anns_tmp.append(elem)


del data['categories'][61:]
data_out = {}
data_out['info'] = data['info']
data_out['licenses'] = data['licenses']
data_out['categories'] = data['categories']
data_out['images'] = imgs_tmp
data_out['annotations'] = anns_tmp

with open('/Users/robertkrutsch/Documents/Code/VD/data/coco2017/annotations/instances_val2017_small.json', 'w') as fp:
    json.dump(data_out, fp)

print("Done")




