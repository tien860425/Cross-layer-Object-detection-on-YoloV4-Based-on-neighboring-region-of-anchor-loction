#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/4
"""

"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from keras import backend as K
from keras.layers import Input
from yolo import YOLO
import cv2
from train import  get_classes, get_anchors
from functools import cmp_to_key
from nets.config import R1, R2

# 輸入圖檔目錄偵測
def detect_img_for_test():
    yolo = YOLO()
    img_path = '../dataset/VOCTest07/JPEGImages'
    paths_list = []

    for parent, _, fileNames in os.walk(img_path):
        for name in fileNames:
            if name.startswith('.'):  # 去除隐藏文件
                continue
            if name.endswith(tuple('jpg')):
                paths_list.append(os.path.join(parent, name))

    for cnt, path in enumerate(paths_list):
        # if cnt in [86,219]:
        #     continue
        image = Image.open(path)
        r_image = yolo.detect_image(image)
        # r_image.save('./xxx{}.png'.format(cnt))
        r_image=np.array(r_image)[...,::-1]
        cv2.imshow(path+'_'+str(cnt),r_image)
        print('image_count:({})'.format(cnt))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    yolo.close_session()




# 輸入圖片路徑檔偵測
def detect_img_for_test1():
    yolo = YOLO()
    img_path = '../dataset/2007_test.txt'

    with open(img_path) as f:
        lines = f.readlines()
    paths_list = []
    paths_list.append('../dataset/VOCdevkit/VOC2007/JPEGImages/000075.jpg')
    paths_list.append('../dataset/VOCdevkit/VOC2007/JPEGImages/000076.jpg')
    paths_list.append('../dataset/VOCdevkit/VOC2007/JPEGImages/000199.jpg')
    paths_list.append('../dataset/VOCdevkit/VOC2007/JPEGImages/000983.jpg')
    paths_list.append('../dataset/VOCdevkit/VOC2007/JPEGImages/001188.jpg')
    paths_list.append('../dataset/VOCdevkit/VOC2007/JPEGImages/002628.jpg')

    for _, line in enumerate(lines):
        paths_list.append(line.split()[0])

    for cnt, path in enumerate(paths_list):
        # if cnt in [86,219]:
        #     continue
        image = Image.open(path)
        r_image = yolo.detect_image(image)
        # r_image.save('./xxx{}.png'.format(cnt))
        r_image = np.array(r_image)[..., ::-1]
        cv2.imshow(path.split('/')[-1] + '_' + str(cnt), r_image)
        print('image_count:({})'.format(cnt))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    yolo.close_session()




def show_img_annotate():  # CelebA #DCNN
    class_names = get_classes('model_data/voc_classes.txt')
    anno_path = '../dataset/2007_test.txt'
    # anno_path=  '../dataset/FDDB_train.txt'
    with open(anno_path) as f:
        lines = f.readlines()
    # with open(train_path) as f:
    #     tls = f.readlines()

    # tls=[]
    for cnt, line in enumerate(lines):
        temp = line.split()
        path = temp[0]
        try:
            image = cv2.imread(path)
            iw, ih = image.shape[:2]
            print(f'imageH:{ih}  imageW:{iw}')
            end = len(temp)

            for i in range(1, end):
                box = temp[i].split(',')
                print(f'x1:{box[0]} y1:{box[1]} x2:{box[2]} y2:{box[3]} ')

                # cv2.imshow('orin_' + str(cnt), image1)
                cv2.rectangle(image, (int(float(box[0])), int(float(box[1]))),
                              (int(float(box[2])), int(float(box[3]))), (0, 255, 0), 2)
                sx, sy = int(float(box[0])) + int((float(box[0])+float(box[2]))/6) ,int(float(box[1]))-15
                cv2.putText(image,class_names[int(float(box[4]))],(sx,sy), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('fix_' + str(cnt), image)
        except:
            print('Error')
        # cv2.imwrite('../dataset/annotationPic/CelebA-{}'.format(str(cnt)+'-'+ path.strip().split('/')[-1] ),image)
        # cv2.imwrite('../dataset/annotationPic1/DCNN/DCNN-{}'.format(str(cnt) + '-' + path.strip().split('/')[-1]),
        #             image)
        print('image_count:({})'.format(cnt))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def get_box_data(annotation_line, input_shape, max_boxes=100):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    # resize image
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    dx = (w-nw)//2
    dy = (h-nh)//2

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image_data = np.array(new_image, np.float32)/255

    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box
        if len(box) > 1:
            box = box.tolist()
            box.sort(key=cmp_to_key(lambda a, b: (a[2] - a[0]) * (a[3] - a[1]) - (b[2] - b[0]) * (b[3] - b[1])))
    # for k, bx in enumerate(box):
    #     top = bx[1]
    #     left = bx[0]
    #     bottom = bx[3]
    #     right = bx[2]
    #     draw = ImageDraw.Draw(new_image)
    #     for j in range(3):  # 畫框
    #         try:
    #             draw.rectangle(
    #                 [left + j, top + j, right - j, bottom - j],
    #                 outline="red")
    #         except NameError:
    #             print('An exception occurred:{}'.format(NameError))            # top = max(0, np.floor(top + 0.5).astype('int32'))

    # r_image= np.array(new_image)[...,::-1]
    #
    # cv2.imshow(line[0], r_image)
    #
    # cv2.waitKey(0)

    return  box_data
def get_y_true(true_boxes, input_shape, anchors, num_classes):
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    # 一共有三個特徵層數
    num_layers = len(anchors) // 3
    # -----------------------------------------------------------#
    #   13x13的特徵層對應的anchor是[142, 110], [192, 243], [459, 401]
    #   26x26的特徵層對應的anchor是[36, 75], [76, 55], [72, 146]
    #   52x52的特徵層對應的anchor是[12, 16], [19, 36], [40, 28]
    # -----------------------------------------------------------#
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # -----------------------------------------------------------#
    #   獲得框的座標和圖片的大小
    # -----------------------------------------------------------#
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    # -----------------------------------------------------------#
    #   通過計算獲得真實框的中心和寬高
    #   中心點(m,n,2) 寬高(m,n,2)
    # -----------------------------------------------------------#
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    # -----------------------------------------------------------#
    #   將真實框歸一化到小數形式
    # -----------------------------------------------------------#
    # 左上右下的框歸一化
    boxes_ltrb = np.zeros(true_boxes[..., 0:4].shape)
    boxes_ltrb[..., 0:2] = boxes_wh / 2 / input_shape[::-1]
    boxes_ltrb[..., 2:4] = boxes_wh / 2 / input_shape[::-1]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    # m為圖片數量，grid_shapes為網格的shape
    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    # -----------------------------------------------------------#
    #   y_true的格式為(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
    # -----------------------------------------------------------#
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 6 + num_classes + 1),
                       dtype='float32') for l in range(num_layers)]

    # -----------------------------------------------------------#
    #   [9,2] -> [1,9,2]
    # -----------------------------------------------------------#
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes

    # -----------------------------------------------------------#
    #   長寬要大於0才有效
    # -----------------------------------------------------------#
    valid_mask = boxes_wh[..., 0] * boxes_wh[..., 1] > 0
    r1 = R1
    r2 = R2

    for b in range(m):
        # 對每一張圖進行處理
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # -----------------------------------------------------------#
        #   [n,2] -> [n,1,2]
        # -----------------------------------------------------------#
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        # -----------------------------------------------------------#
        #   計算所有真實框和先驗框的交並比
        #   intersect_area  [n,9]
        #   box_area        [n,1]
        #   anchor_area     [1,9]
        #   iou             [n,9]
        # -----------------------------------------------------------#
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]

        iou = intersect_area / (box_area + anchor_area - intersect_area)
        # -----------------------------------------------------------#
        #   維度是[n,]
        # -----------------------------------------------------------#
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            # -----------------------------------------------------------#
            #   找到每個真實框所屬的特徵層
            # -----------------------------------------------------------#
            for l in range(num_layers):
                # -----------------------------------------------------------#
                #   floor用於向下取整，找到真實框所屬的特徵層對應的x、y軸座標
                # -----------------------------------------------------------#
                i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                # effective area
                iefp = np.floor((true_boxes[b, t, 0] + true_boxes[b, t, 2] * r1) * grid_shapes[l][1]).astype(
                    'int32')
                iefn = np.floor((true_boxes[b, t, 0] - true_boxes[b, t, 2] * r1) * grid_shapes[l][1]).astype(
                    'int32')
                if iefp > grid_shapes[l][1] - 1:
                    iefp = grid_shapes[l][1] - 1
                if iefn < 0:
                    iefn = 0
                jefp = np.floor((true_boxes[b, t, 1] + true_boxes[b, t, 3] * r1) * grid_shapes[l][0]).astype(
                    'int32')
                jefn = np.floor((true_boxes[b, t, 1] - true_boxes[b, t, 3] * r1) * grid_shapes[l][0]).astype(
                    'int32')
                if jefp > grid_shapes[l][0] - 1:
                    jefp = grid_shapes[l][0] - 1
                if jefn < 0:
                    jefn = 0

                # # ignor area
                iprp = np.floor((true_boxes[b, t, 0] + true_boxes[b, t, 2] * r2) * grid_shapes[l][1]).astype(
                    'int32')
                iprn = np.floor((true_boxes[b, t, 0] - true_boxes[b, t, 2] * r2) * grid_shapes[l][1]).astype(
                    'int32')
                jprp = np.floor((true_boxes[b, t, 1] + true_boxes[b, t, 3] * r2) * grid_shapes[l][0]).astype(
                    'int32')
                jprn = np.floor((true_boxes[b, t, 1] - true_boxes[b, t, 3] * r2) * grid_shapes[l][0]).astype(
                    'int32')
                if iprp > grid_shapes[l][1] - 1:
                    iprp = grid_shapes[l][1] - 1
                if iprn < 0:
                    iprn = 0
                if jprp > grid_shapes[l][0] - 1:
                    jprp = grid_shapes[l][0] - 1
                if jprn < 0:
                    jprn = 0
                if n in anchor_mask[l]:

                    box_ltrb = boxes_ltrb[b, t, 0:4] * np.concatenate(
                        (grid_shapes[l][::-1], grid_shapes[l][::-1]),
                        axis=-1)

                    # -----------------------------------------------------------#
                    #   k指的的當前這個特徵層的第k個先驗框
                    # -----------------------------------------------------------#
                    k = anchor_mask[l].index(n)
                    # -----------------------------------------------------------#
                    #   c指的是當前這個真實框的種類
                    # -----------------------------------------------------------#
                    for a in range(3):
                        c = true_boxes[b, t, 4].astype('int32')
                        OuterArea = y_true[l][b, jprn:jprp + 1, iprn:iprp + 1, a:a + 1, :]
                        mask_outerArea = OuterArea[..., 5] <= 0
                        effectiveArea = y_true[l][b, jefn:jefp + 1, iefn:iefp + 1, a:a + 1, :]
                        mask_effectiveArea = effectiveArea[..., 4] <= 0
                        if a == k:
                            grid_y = np.tile(np.reshape(np.arange(jprn, stop=jprp + 1), [-1, 1, 1]),
                                             [1, iprp - iprn + 1, 1])
                            grid_x = np.tile(np.reshape(np.arange(iprn, stop=iprp + 1), [1, -1, 1]),
                                             [jprp - jprn + 1, 1, 1])
                            grid = np.concatenate([grid_x, grid_y], axis=-1)
                            boxes_xy = (true_boxes[b, t, 0:2] * grid_shapes[l][::-1] - np.array(
                                [i, j]))  # # 中心點偏移 point(cell)的左上角
                            # 以point中心再校正ltrb
                            boxes_lt = (box_ltrb[0:2] - (boxes_xy - 0.5) + (
                                        grid[..., 0:2] - np.array([i, j]))) / \
                                       grid_shapes[
                                           l][::-1]
                            boxes_rb = (box_ltrb[2:4] + (boxes_xy - 0.5) - (
                                        grid[..., 0:2] - np.array([i, j]))) / \
                                       grid_shapes[
                                           l][::-1]
                            boxes_lt = np.expand_dims(boxes_lt, -2)
                            boxes_rb = np.expand_dims(boxes_rb, -2)
                            tempProj = OuterArea.copy()
                            tempProj[..., 0:4] = np.concatenate([boxes_lt, boxes_rb], axis=-1)
                            tempProj[..., 5:6] = (t + 1)  # object 編號
                            tempProj[..., 6 + c] = 1
                            OuterArea[mask_outerArea] = tempProj[mask_outerArea]
                            y_true[l][b, jprn:jprp + 1, iprn:iprp + 1, a:a + 1, :] = OuterArea
                            tempProj = effectiveArea.copy()
                            tempProj[..., 4:5] = (t + 1)  # 有效區
                            effectiveArea[mask_effectiveArea] = tempProj[mask_effectiveArea]

                            y_true[l][b, jefn:jefp + 1, iefn:iefp + 1, a:a + 1, :] = effectiveArea
                            y_true[l][b, j, i, a:a + 1, -1] = 1
                            y_true[l][b, j, i, k, 4] = t + 1
                        else:
                            tempProj = OuterArea.copy()
                            tempProj[..., 5:6] = -1  # ignore
                            OuterArea[mask_outerArea] = tempProj[mask_outerArea]
                            y_true[l][b, jprn:jprp + 1, iprn:iprp + 1, a:a + 1, :] = OuterArea

                            tempProj = effectiveArea.copy()
                            tempProj[..., 4:5] = -1  # ignore
                            effectiveArea[mask_effectiveArea] = tempProj[mask_effectiveArea]

                            y_true[l][b, jefn:jefp + 1, iefn:iefp + 1, a:a + 1, :] = effectiveArea

    return y_true
def detect_img_for_test3():
    class_names = get_classes('model_data/voc_classes.txt')
    # 畫框設置不同的顏色
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    yolo = YOLO(3)
    img_path = '../dataset/2007_test.txt'

    with open(img_path) as f:
        lines = f.readlines()
    paths_list = []
    paths_list.append('../dataset/VOCdevkit/VOC2007/JPEGImages/000075.jpg 102,83,222,348,11')
    paths_list.append('../dataset/VOCdevkit/VOC2007/JPEGImages/000076.jpg 63,78,265,375,14 257,75,448,375,14 362,130,446,261,14 472,157,500,343,14')
    paths_list.append('../dataset/VOCdevkit/VOC2007/JPEGImages/000199.jpg 82,68,383,270,2')
    paths_list.append('../dataset/VOCdevkit/VOC2007/JPEGImages/000983.jpg 36,70,288,486,15')
    paths_list.append('../dataset/VOCdevkit/VOC2007/JPEGImages/001188.jpg 288,95,318,124,3 344,63,377,87,3 325,97,359,123,3 376,96,408,125,3 404,64,445,86,3 ')
    paths_list.append('../dataset/VOCdevkit/VOC2007/JPEGImages/002628.jpg 235,57,351,306,3 172,78,284,280,3 64,168,121,267,3')

    for _, line in enumerate(lines):
        paths_list.append(line)

    for cnt, linea in enumerate(paths_list):        # if cnt in [86,219]:
        #     continue
        temp = linea.split()
        path = temp[0]
        image = Image.open(path)
        # image.show()
        boxes=get_box_data(linea, (416, 416))
        boxes=np.expand_dims(boxes,axis=0)
        y_true=get_y_true(boxes,(416, 416),
                                   get_anchors('model_data/yolo_anchors_416.txt'),20)
        r_image, imcount = yolo.detect_image_3(image,y_true)
        for l in range(3):
            for i in range(len(r_image[l])):
                for j in range(len(r_image[l][i])):
            # r_image.save('./xxx{}.png'.format(cnt))
                    r_image_ = np.array(r_image[l][i][j])[..., ::-1]
                    # cv2.imshow(path.split('/')[-1] + '_' + str(cnt), r_image)
                    cv2.imshow('Head2_predict_layer ' + str(l+1) + "_" + str(i+1)+ "_" + str(j+1) , r_image_)

        end = len(temp)
        font = ImageFont.truetype(font='font/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        for i in range(1, end):
            box = temp[i].split(',')
            left,top, right, bottom, c = map(lambda x:int(float(x)),box)
            print(f'x1:{box[0]} y1:{box[1]} x2:{box[2]} y2:{box[3]} ')
            predicted_class = class_names[c]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # 畫框框
            label = '{}'.format(predicted_class)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        image=np.array(image)[..., ::-1]
        # print('image_count:({})'.format(cnt))
        #------------------------------
        # for anotation
        # image = cv2.imread(path)
        # end = len(temp)
        # # font = ImageFont.truetype(font='font/simhei.ttf',
        # #                           size=np.floor(3e-2 * image.size[0] + 0.5).astype('int32'))
        # for i in range(1, end):
        #     box = temp[i].split(',')
        #     print(f'x1:{box[0]} y1:{box[1]} x2:{box[2]} y2:{box[3]} ')
        #
        #     # cv2.imshow('orin_' + str(cnt), image1)
        #     cv2.rectangle(image, (int(float(box[0])), int(float(box[1]))),
        #                   (int(float(box[2])), int(float(box[3]))), colors[int(float(box[4]))], 2)
        #     sx, sy = int(float(box[0])) + int((float(box[0])+float(box[2]))/6) ,int(float(box[1]))-15
        #     cv2.putText(image,class_names[int(float(box[4]))],(sx,sy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Head4_original_' + str(cnt), image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    yolo.close_session()


# show picture with labels___wider or FDDB
def show_img_annotate_det():  # wider or FDDB
    # yolo = YOLO()
    # anno_path= '../dataset/Wider_train.txt'
    anno_path = '../dataset/FDDB_train.txt'

    with open(anno_path) as f:
        lines = f.readlines()

    # tls=[]
    for cnt, line in enumerate(lines):
        # if cnt in [86,219]:
        #     continue
        # print('process',cnt,'-' ,len(lines))
        # temp1=tls[cnt].split()
        # path1=temp1[0]
        # box1=temp1[1].split(',')
        # landmark1=temp1[2].split(',')
        # image1=cv2.imread(path1)
        # r_image1 = Image.open(path1)
        temp = line.split()
        path = temp[0]
        try:
            image = cv2.imread(path)
            iw, ih = image.shape[:2]
            print(f'imageH:{ih}  imageW:{iw}')
            end = len(temp)

            for i in range(1, end):
                box = temp[i].split(',')
                print(f'x1:{box[0]} y1:{box[1]} x2:{box[2]} y2:{box[3]} ')
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                # for i in range(len(landmark)//2):
                #     cv2.circle(image,(int(landmark[2*i]),int(landmark[2*i+1])),3,(0, 255, 255), 2)
            cv2.imshow(path.split('/')[-1] + str(cnt), image)
        except:
            print('Error')
        cv2.imwrite('../dataset/annotationPic/FDDB/FDDB-{}'.format(str(cnt) + '-' + path.strip().split('/')[-1]),
                    image)
        print('image_count:({})'.format(cnt))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
    # yolo.close_session()



if __name__ == '__main__':
    # show_img_annotate_dir()
    detect_img_for_test3()
    # show_img_annotate()