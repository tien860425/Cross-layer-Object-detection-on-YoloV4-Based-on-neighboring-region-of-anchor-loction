import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam

from nets.loss import yolo_loss
from nets.yolo4 import yolo_body
from utils.utils import (WarmUpCosineDecayScheduler, get_random_data,
                         get_random_data_with_Mosaic, rand)

from matplotlib import pyplot
from nets.config import R1, R2
# ---------------------------------------------------#
#   獲得類和先驗框
# ---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


# ---------------------------------------------------#
#   訓練數據生成器
# ---------------------------------------------------#
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, mosaic=False, random=True):
    n = len(annotation_lines)
    i = 0
    flag = True
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            if mosaic:
                if flag and (i + 4) < n:
                    image, box = get_random_data_with_Mosaic(annotation_lines[i:i + 4], input_shape)
                    i = (i + 4) % n
                else:
                    image, box = get_random_data(annotation_lines[i], input_shape, random=random)
                    i = (i + 1) % n
                flag = bool(1 - flag)
            else:
                image, box = get_random_data(annotation_lines[i], input_shape, random=random)
                i = (i + 1) % n
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true, true_box  = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true, true_box], np.zeros(batch_size)


# ---------------------------------------------------#
#   讀入xml檔，並輸出y_true
# ---------------------------------------------------#
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
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
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 6 + num_classes),
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
    r1 = R1  # 投影區的中心 25%
    r2 = R2  # 50%

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

                # ignor area
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

                    box_ltrb = boxes_ltrb[b, t, 0:4] * np.concatenate((grid_shapes[l][::-1], grid_shapes[l][::-1]),
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
                            boxes_lt = (box_ltrb[0:2] - (boxes_xy - 0.5) + (grid[..., 0:2] - np.array([i, j]))) / \
                                       grid_shapes[
                                           l][::-1]
                            boxes_rb = (box_ltrb[2:4] + (boxes_xy - 0.5) - (grid[..., 0:2] - np.array([i, j]))) / \
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
                            tempProj[..., 4:5] = (t + 1)  # object 編號
                            effectiveArea[mask_effectiveArea] = tempProj[mask_effectiveArea]

                            y_true[l][b, jefn:jefp + 1, iefn:iefp + 1, a:a + 1, :] = effectiveArea

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

    return y_true, true_boxes[...,:4]



# def preprocess_true_boxes_test(true_boxes, input_shape, anchors, num_classes):
#     assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
#     # 一共有三個特徵層數
#     num_layers = len(anchors) // 3
#     # -----------------------------------------------------------#
#     #   13x13的特徵層對應的anchor是[142, 110], [192, 243], [459, 401]
#     #   26x26的特徵層對應的anchor是[36, 75], [76, 55], [72, 146]
#     #   52x52的特徵層對應的anchor是[12, 16], [19, 36], [40, 28]
#     # -----------------------------------------------------------#
#     anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
#
#     # -----------------------------------------------------------#
#     #   獲得框的座標和圖片的大小
#     # -----------------------------------------------------------#
#     true_boxes = np.array(true_boxes, dtype='float32')
#     input_shape = np.array(input_shape, dtype='int32')
#     # -----------------------------------------------------------#
#     #   通過計算獲得真實框的中心和寬高
#     #   中心點(m,n,2) 寬高(m,n,2)
#     # -----------------------------------------------------------#
#     boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
#     boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
#     # -----------------------------------------------------------#
#     #   將真實框歸一化到小數形式
#     # -----------------------------------------------------------#
#     # 左上右下的框歸一化
#     boxes_ltrb = np.zeros(true_boxes[..., 0:4].shape)
#     boxes_ltrb[..., 0:2] = boxes_wh / 2 / input_shape[::-1]
#     boxes_ltrb[..., 2:4] = boxes_wh / 2 / input_shape[::-1]
#     true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
#     true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]
#
#     # m為圖片數量，grid_shapes為網格的shape
#     m = true_boxes.shape[0]
#     grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
#     # -----------------------------------------------------------#
#     #   y_true的格式為(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
#     # -----------------------------------------------------------#
#     y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 6 + num_classes+1),
#                        dtype='float32') for l in range(num_layers)]
#
#     # -----------------------------------------------------------#
#     #   [9,2] -> [1,9,2]
#     # -----------------------------------------------------------#
#     anchors = np.expand_dims(anchors, 0)
#     anchor_maxes = anchors / 2.
#     anchor_mins = -anchor_maxes
#
#     # -----------------------------------------------------------#
#     #   長寬要大於0才有效
#     # -----------------------------------------------------------#
#     valid_mask = boxes_wh[..., 0] * boxes_wh[..., 1] > 0
#     r1 = R1  # 投影區的中心 25%
#     r2 = R2  # 50%
#
#     for b in range(m):
#         # 對每一張圖進行處理
#         wh = boxes_wh[b, valid_mask[b]]
#         if len(wh) == 0: continue
#         # -----------------------------------------------------------#
#         #   [n,2] -> [n,1,2]
#         # -----------------------------------------------------------#
#         wh = np.expand_dims(wh, -2)
#         box_maxes = wh / 2.
#         box_mins = -box_maxes
#
#         # -----------------------------------------------------------#
#         #   計算所有真實框和先驗框的交並比
#         #   intersect_area  [n,9]
#         #   box_area        [n,1]
#         #   anchor_area     [1,9]
#         #   iou             [n,9]
#         # -----------------------------------------------------------#
#         intersect_mins = np.maximum(box_mins, anchor_mins)
#         intersect_maxes = np.minimum(box_maxes, anchor_maxes)
#         intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
#         intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
#
#         box_area = wh[..., 0] * wh[..., 1]
#         anchor_area = anchors[..., 0] * anchors[..., 1]
#
#         iou = intersect_area / (box_area + anchor_area - intersect_area)
#         # -----------------------------------------------------------#
#         #   維度是[n,]
#         # -----------------------------------------------------------#
#         best_anchor = np.argmax(iou, axis=-1)
#
#         for t, n in enumerate(best_anchor):
#             # -----------------------------------------------------------#
#             #   找到每個真實框所屬的特徵層
#             # -----------------------------------------------------------#
#             for l in range(num_layers):
#                 # -----------------------------------------------------------#
#                 #   floor用於向下取整，找到真實框所屬的特徵層對應的x、y軸座標
#                 # -----------------------------------------------------------#
#                 i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
#                 j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
#                 # effective area
#                 iefp = np.floor((true_boxes[b, t, 0] + true_boxes[b, t, 2] * r1) * grid_shapes[l][1]).astype(
#                     'int32')
#                 iefn = np.floor((true_boxes[b, t, 0] - true_boxes[b, t, 2] * r1) * grid_shapes[l][1]).astype(
#                     'int32')
#                 if iefp > grid_shapes[l][1] - 1:
#                     iefp = grid_shapes[l][1] - 1
#                 if iefn < 0:
#                     iefn = 0
#                 jefp = np.floor((true_boxes[b, t, 1] + true_boxes[b, t, 3] * r1) * grid_shapes[l][0]).astype(
#                     'int32')
#                 jefn = np.floor((true_boxes[b, t, 1] - true_boxes[b, t, 3] * r1) * grid_shapes[l][0]).astype(
#                     'int32')
#                 if jefp > grid_shapes[l][0] - 1:
#                     jefp = grid_shapes[l][0] - 1
#                 if jefn < 0:
#                     jefn = 0
#
#                 # # ignor area
#                 iprp = np.floor((true_boxes[b, t, 0] + true_boxes[b, t, 2] * r2) * grid_shapes[l][1]).astype(
#                     'int32')
#                 iprn = np.floor((true_boxes[b, t, 0] - true_boxes[b, t, 2] * r2) * grid_shapes[l][1]).astype(
#                     'int32')
#                 jprp = np.floor((true_boxes[b, t, 1] + true_boxes[b, t, 3] * r2) * grid_shapes[l][0]).astype(
#                     'int32')
#                 jprn = np.floor((true_boxes[b, t, 1] - true_boxes[b, t, 3] * r2) * grid_shapes[l][0]).astype(
#                     'int32')
#                 if iprp > grid_shapes[l][1] - 1:
#                     iprp = grid_shapes[l][1] - 1
#                 if iprn < 0:
#                     iprn = 0
#                 if jprp > grid_shapes[l][0] - 1:
#                     jprp = grid_shapes[l][0] - 1
#                 if jprn < 0:
#                     jprn = 0
#                 if n in anchor_mask[l]:
#
#                     box_ltrb = boxes_ltrb[b, t, 0:4] * np.concatenate((grid_shapes[l][::-1], grid_shapes[l][::-1]),
#                                                                           axis=-1)
#
#                     # -----------------------------------------------------------#
#                     #   k指的的當前這個特徵層的第k個先驗框
#                     # -----------------------------------------------------------#
#                     k = anchor_mask[l].index(n)
#                     # -----------------------------------------------------------#
#                     #   c指的是當前這個真實框的種類
#                     # -----------------------------------------------------------#
#                     for a in range(3):
#                         c = true_boxes[b, t, 4].astype('int32')
#                         OuterArea = y_true[l][b, jprn:jprp + 1, iprn:iprp + 1, a:a + 1, :]
#                         mask_outerArea = OuterArea[..., 5] <= 0
#                         effectiveArea = y_true[l][b, jefn:jefp + 1, iefn:iefp + 1, a:a + 1, :]
#                         mask_effectiveArea = effectiveArea[..., 4] <= 0
#                         if a == k:
#                             grid_y = np.tile(np.reshape(np.arange(jprn, stop=jprp + 1), [-1, 1, 1]),
#                                              [1, iprp - iprn + 1, 1])
#                             grid_x = np.tile(np.reshape(np.arange(iprn, stop=iprp + 1), [1, -1, 1]),
#                                              [jprp - jprn + 1, 1, 1])
#                             grid = np.concatenate([grid_x, grid_y], axis=-1)
#                             boxes_xy = (true_boxes[b, t, 0:2] * grid_shapes[l][::-1] - np.array(
#                                 [i, j]))  # # 中心點偏移 point(cell)的左上角
#                             # 以point中心再校正ltrb
#                             boxes_lt = (box_ltrb[0:2] - (boxes_xy - 0.5) + (grid[..., 0:2] - np.array([i, j]))) / \
#                                        grid_shapes[
#                                            l][::-1]
#                             boxes_rb = (box_ltrb[2:4] + (boxes_xy - 0.5) - (grid[..., 0:2] - np.array([i, j]))) / \
#                                        grid_shapes[
#                                            l][::-1]
#                             boxes_lt = np.expand_dims(boxes_lt, -2)
#                             boxes_rb = np.expand_dims(boxes_rb, -2)
#                             tempProj = OuterArea.copy()
#                             tempProj[..., 0:4] = np.concatenate([boxes_lt, boxes_rb], axis=-1)
#                             tempProj[..., 5:6] = (t + 1)  # object 編號
#                             tempProj[..., 6 + c] = 1
#                             OuterArea[mask_outerArea] = tempProj[mask_outerArea]
#                             y_true[l][b, jprn:jprp + 1, iprn:iprp + 1, a:a + 1, :] = OuterArea
#                             tempProj = effectiveArea.copy()
#                             tempProj[..., 4:5] = (t + 1)  # 有效區
#                             effectiveArea[mask_effectiveArea] = tempProj[mask_effectiveArea]
#
#                             y_true[l][b, jefn:jefp + 1, iefn:iefp + 1, a:a + 1, :] = effectiveArea
#                             y_true[l][b, j, i, a:a + 1, -1]=1
#                             y_true[l][b, j, i, k, 4] = t + 1
#                         else:
#                             tempProj = OuterArea.copy()
#                             tempProj[..., 5:6] = -1  # ignore
#                             OuterArea[mask_outerArea] = tempProj[mask_outerArea]
#                             y_true[l][b, jprn:jprp + 1, iprn:iprp + 1, a:a + 1, :] = OuterArea
#
#                             tempProj = effectiveArea.copy()
#                             tempProj[..., 4:5] = -1  # ignore
#                             effectiveArea[mask_effectiveArea] = tempProj[mask_effectiveArea]
#
#                             y_true[l][b, jefn:jefp + 1, iefn:iefp + 1, a:a + 1, :] = effectiveArea
#
#     return y_true, true_boxes[...,:4]


# ----------------------------------------------------#
#   檢測精度mAP和pr曲線計算參考視頻
#   https://www.bilibili.com/video/BV1zE411u7Vw
# ----------------------------------------------------#
if __name__ == "__main__":
    # ----------------------------------------------------#
    #   獲得圖片路徑和標籤
    # ----------------------------------------------------#
    # annotation_path = '../dataset/mscocotrain.txt'
    annotation_path = '../dataset/2007_train.txt'
    annotation_path1 = '../dataset/2012_train.txt'
    # annotation_path2 = '../dataset/mscoco_train20.txt'
    # 獲取classes和anchor的位置
    # classes_path = 'model_data/coco_classes.txt'
    # anchors_path = 'model_data/yolo_anchors.txt'
    # ------------------------------------------------------#
    #   訓練後的模型保存的位置，保存在logs資料夾裡面
    # ------------------------------------------------------#
    log_dir = 'logs/voc/'
    # ----------------------------------------------------#
    #   classes和anchor的路徑，非常重要
    #   訓練前一定要修改classes_path，使其對應自己的資料集
    # ----------------------------------------------------#
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors_416.txt'
    # classes_path = 'model_data/coco_classes.txt'
    # anchors_path = 'model_data/yolo_anchors.txt'

    # ------------------------------------------------------#
    #   訓練自己的資料集時提示維度不匹配正常
    #   預測的東西都不一樣了自然維度不匹配
    # ------------------------------------------------------#
    weights_path = 'model_data/yolo4_voc_weights.h5'
    # ------------------------------------------------------#
    #   訓練用圖片大小
    #   一般在416x416和608x608選擇
    # ------------------------------------------------------#
    input_shape = (416, 416)
    # ------------------------------------------------------#
    #   是否對損失進行歸一化，用於改變loss的大小
    #   用於決定計算最終loss是除上batch_size還是除上正樣本數量
    # ------------------------------------------------------#
    normalize = True

    # -----------------------------------------------------#
    #   獲取classes和anchor
    # -----------------------------------------------------#
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    # ------------------------------------------------------#
    #   一共有多少類和多少先驗框
    # ------------------------------------------------------#
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # ------------------------------------------------------#
    #   Yolov4的tricks應用
    #   mosaic 馬賽克資料增強 True or False
    #   實際測試時mosaic資料增強並不穩定，所以默認為False
    #   Cosine_scheduler 余弦退火學習率 True or False
    #   label_smoothing 標籤平滑 0.01以下一般 如0.01、0.005
    # ------------------------------------------------------#
    mosaic = False
    Cosine_scheduler = False
    label_smoothing = 0

    K.clear_session()
    # ------------------------------------------------------#
    #   創建yolo模型
    # ------------------------------------------------------#
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    print('Create YOLOv4 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    model_body = yolo_body(image_input, num_anchors // 3, num_classes)

    # ------------------------------------------------------#
    #   載入預訓練權重
    # ------------------------------------------------------#
    print('Load weights {}.'.format(weights_path))
    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

    # ------------------------------------------------------#
    #   在這個地方設置損失，將網路的輸出結果傳入loss函數
    #   把整個模型的輸出作為loss
    # ------------------------------------------------------#
    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 6)) for l in range(3)]
    y_true_box = Input(shape=(100, 4), dtype='float32')
    loss_input = [*model_body.output, *y_true, y_true_box]
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5,
                                   'label_smoothing': label_smoothing, 'normalize': normalize})(loss_input)

    model = Model([model_body.input, *y_true,y_true_box], model_loss)

    # -------------------------------------------------------------------------------#
    #   訓練參數的設置
    #   logging表示tensorboard的保存位址
    #   checkpoint用於設置權值保存的細節，period用於修改多少epoch保存一次
    #   reduce_lr用於設置學習率下降的方式
    #   early_stopping用於設定早停，val_loss多次不下降自動結束訓練，表示模型基本收斂
    # -------------------------------------------------------------------------------#
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # ----------------------------------------------------------------------#
    #   驗證集的劃分在train.py代碼裡面進行
    #   2007_test.txt和2007_val.txt裡面沒有內容是正常的。訓練不會使用到。
    #   當前劃分方式下，驗證集和訓練集的比例為1:9
    # ----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    with open(annotation_path1) as f:
        lines1 = f.readlines()
    lines=lines+lines1
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    freeze_layers = 249
    for i in range(freeze_layers): model_body.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))

    # ------------------------------------------------------#
    #   主幹特徵提取網路特徵通用，凍結訓練可以加快訓練速度
    #   也可以在訓練初期防止權值被破壞。
    #   Init_Epoch為起始世代
    #   Freeze_Epoch為凍結訓練的世代
    #   Epoch總訓練世代
    #   提示OOM或者顯存不足請調小Batch_size
    # ------------------------------------------------------#
    if True:
        Init_epoch = 0
        Freeze_epoch = 100
        batch_size = 1
        learning_rate_base = 1e-3

        if Cosine_scheduler:
            # 預熱期
            warmup_epoch = int((Freeze_epoch - Init_epoch) * 0.2)
            # 總共的步長
            total_steps = int((Freeze_epoch - Init_epoch) * num_train / batch_size)
            # 預熱步長
            warmup_steps = int(warmup_epoch * num_train / batch_size)
            # 學習率
            reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                                   total_steps=total_steps,
                                                   warmup_learning_rate=1e-4,
                                                   warmup_steps=warmup_steps,
                                                   hold_base_rate_steps=num_train,
                                                   min_learn_rate=1e-6
                                                   )
            model.compile(optimizer=Adam(), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        else:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
            model.compile(optimizer=Adam(learning_rate_base), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        history=model.fit_generator(
            data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, mosaic=mosaic,
                            random=True),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes,
                                           mosaic=False, random=False),
            validation_steps=max(1, num_val // batch_size),
            epochs=Freeze_epoch,
            initial_epoch=Init_epoch,
            callbacks=[logging, checkpoint, reduce_lr])#, early_stopping])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')
        pyplot.plot(history.history['loss'])
        pyplot.plot(history.history['val_loss'])
        pyplot.title('model train and validation Freeze BackBone')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper right')
        pyplot.savefig(log_dir + "model train and validation Freeze BackBone.png")

    for i in range(freeze_layers): model_body.layers[i].trainable = True

    if False:
        Freeze_epoch = 100
        Epoch = 200
        batch_size = 16
        learning_rate_base = 1e-4

        if Cosine_scheduler:
            # 預熱期
            warmup_epoch = int((Epoch - Freeze_epoch) * 0.2)
            # 總共的步長
            total_steps = int((Epoch - Freeze_epoch) * num_train / batch_size)
            # 預熱步長
            warmup_steps = int(warmup_epoch * num_train / batch_size)
            # 學習率
            reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                                   total_steps=total_steps,
                                                   warmup_learning_rate=1e-5,
                                                   warmup_steps=warmup_steps,
                                                   hold_base_rate_steps=num_train // 2,
                                                   min_learn_rate=1e-6
                                                   )
            model.compile(optimizer=Adam(), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        else:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
            model.compile(optimizer=Adam(learning_rate_base), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        history=model.fit_generator(
            data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, mosaic=mosaic,
                           random=True),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes,
                                           mosaic=False, random=False),
            validation_steps=max(1, num_val // batch_size),
            epochs=Epoch,
            initial_epoch=Freeze_epoch,
            callbacks=[logging, checkpoint, reduce_lr])#, early_stopping])
        model.save_weights(log_dir + 'last1.h5')
        pyplot.plot(history.history['loss'])
        pyplot.plot(history.history['val_loss'])
        pyplot.title('model train and validation unFreeze All')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper right')
        pyplot.savefig(log_dir + "model train and validation unFreeze All.png")


