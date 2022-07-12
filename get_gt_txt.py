#----------------------------------------------------#
#   獲取測試集的ground-truth
#   具體視頻教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
import sys
import os
import glob
import xml.etree.ElementTree as ET

'''
！！！！！！！！！！！！！注意事項！！！！！！！！！！！！！
# 這一部分是當xml有無關的類的時候，下方有代碼可以進行篩選！
'''
#---------------------------------------------------#
#   獲得類
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

image_ids = open('../dataset/VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/ground-truth"):
    os.makedirs("./input/ground-truth")

for image_id in image_ids:
    with open("./input/ground-truth/"+image_id+".txt", "w") as new_f:
        root = ET.parse("../dataset/VOCdevkit/VOC2007/Annotations/"+image_id+".xml").getroot()
        for obj in root.findall('object'):
            difficult_flag = False
            if obj.find('difficult')!=None:
                difficult = obj.find('difficult').text
                if int(difficult)==1:
                    difficult_flag = True
            obj_name = obj.find('name').text
            '''
            ！！！！！！！！！！！！注意事項！！！！！！！！！！！！
            # 這一部分是當xml有無關的類的時候，可以取消下面代碼的注釋
            # 利用對應的classes.txt來進行篩選！！！！！！！！！！！！
            '''
            # classes_path = 'model_data/voc_classes.txt'
            # class_names = get_classes(classes_path)
            # if obj_name not in class_names:
            #     continue

            bndbox = obj.find('bndbox')
            left = bndbox.find('xmin').text
            top = bndbox.find('ymin').text
            right = bndbox.find('xmax').text
            bottom = bndbox.find('ymax').text

            if difficult_flag:
                new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
            else:
                new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

print("Conversion completed!")

