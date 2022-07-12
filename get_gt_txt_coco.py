#----------------------------------------------------#
#   获取测试集的ground-truth
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
import sys
import os
import glob
import xml.etree.ElementTree as ET
from train import  get_classes

img_path = '../dataset/mscocoval.txt'
class_names = get_classes('model_data/coco_classes.txt')
if not os.path.exists("./input_coco"):
    os.makedirs("./input_coco")
if not os.path.exists("./input_coco/ground-truth"):
    os.makedirs("./input_coco/ground-truth")
with open(img_path) as f:
    lines = f.readlines()
    for cnt, line in enumerate(lines):
        # if cnt in [86,219]:
        #     continue
        temp = line.split()
        image_id = temp[0].split('/')[-1].split('.')[0]
        end = len(temp)
        with open("./input_coco/ground-truth/" + image_id + ".txt", "w") as new_f:
            for i in range(1, end):
                box = temp[i].split(',')
                left,top, right, bottom, c = map(lambda x:int(float(x)),box)
                class_name = class_names[c]
                new_f.write("%s %s %s %s %s\n" % (class_name, box[0], box[1],box[2],box[3]))

print("Conversion completed!")
