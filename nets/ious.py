from keras import backend as K
import tensorflow as tf
import math
import numpy as np
def box_ciou(b1, b2):
    """
    輸入為：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    返回為：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    #-----------------------------------------------------------#
    #   求出預測框左上角右下角
    #   b1_mins     (batch, feat_w, feat_h, anchor_num, 2)
    #   b1_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    #-----------------------------------------------------------#
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # -----------------------------------------------------------#
    #   求出真實框左上角右下角
    #   b2_mins     (batch, feat_w, feat_h, anchor_num, 2)
    #   b2_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    # -----------------------------------------------------------#
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # -----------------------------------------------------------#
    #   求真實框和預測框所有的iou
    #   iou         (batch, feat_w, feat_h, anchor_num)
    # -----------------------------------------------------------#
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / K.maximum(union_area, K.epsilon())

    # -----------------------------------------------------------#
    #   計算中心的差距
    #   center_distance (batch, feat_w, feat_h, anchor_num)
    # -----------------------------------------------------------#
    center_distance = K.sum(K.square(b1_xy - b2_xy), axis=-1)
    #
    enclose_mins = K.minimum(b1_mins, b2_mins)
    enclose_maxes = K.maximum(b1_maxes, b2_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    ##-----------------------------------------------------------#
    #   計算對角線距離
    #   enclose_diagonal (batch, feat_w, feat_h, anchor_num)
    #-----------------------------------------------------------#
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    ciou = iou - 1.0 * (center_distance) / K.maximum(enclose_diagonal, K.epsilon())

    v = 4 * K.square(tf.math.atan2(b1_wh[..., 0], K.maximum(b1_wh[..., 1], K.epsilon())) - tf.math.atan2(b2_wh[..., 0],
                                                                                                         K.maximum(
                                                                                                             b2_wh[
                                                                                                                 ..., 1],
                                                                                                             K.epsilon()))) / (
                    math.pi * math.pi)
    alpha = v / K.maximum((1.0 - iou + v), K.epsilon())
    ciou = ciou - alpha * v

    ciou = K.expand_dims(ciou, -1)
    return ciou

def b_ciou(b1,b2): #numpy version
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # 求出真實框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真實框和預測框所有的iou
    intersect_mins = np.maximum(b1_mins, b2_mins)
    intersect_maxes = np.minimum(b1_maxes, b2_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / np.maximum(union_area, 0.0000001)

    # 計算中心的差距
    center_distance = np.sum(np.square(b1_xy - b2_xy), axis=-1)
    # 找到包裹兩個框的最小框的左上角和右下角
    enclose_mins = np.minimum(b1_mins, b2_mins)
    enclose_maxes = np.maximum(b1_maxes, b2_maxes)
    enclose_wh = np.maximum(enclose_maxes - enclose_mins, 0.0)
    # 計算對角線距離
    enclose_diagonal = np.sum(np.square(enclose_wh), axis=-1)
    ciou = iou - 1.0 * (center_distance) / np.maximum(enclose_diagonal, 0.0000000001)

    v = 4 * np.square(np.arctan(b1_wh[..., 0] / np.maximum(b1_wh[..., 1], 0.0000000001)) - np.arctan(b2_wh[..., 0] /
            np.maximum(b2_wh[..., 1],0.0000001))) / (math.pi * math.pi)
    alpha = v / np.maximum((1.0 - iou + v), 0.0000001)
    ciou = ciou - alpha * v
    return ciou[0]
def b_iou(b1,b2): #numpy version
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # 求出真實框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真實框和預測框所有的iou
    intersect_mins = np.maximum(b1_mins, b2_mins)
    intersect_maxes = np.minimum(b1_maxes, b2_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / np.maximum(union_area, 0.0000001)


    return iou

def bbox_ciou(boxes1, boxes2):
    '''
    計算ciou = iou - p2/c2 - av
    :param boxes1: (8, 13, 13, 3, 4)   pred_xywh
    :param boxes2: (8, 13, 13, 3, 4)   label_xywh
    :return:

    舉例時假設pred_xywh和label_xywh的shape都是(1, 4)
    '''

    # 變成左上角座標、右下角座標
    boxes1_x0y0x1y1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_x0y0x1y1 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    '''
    逐個位置比較boxes1_x0y0x1y1[..., :2]和boxes1_x0y0x1y1[..., 2:]，即逐個位置比較[x0, y0]和[x1, y1]，小的留下。
    比如留下了[x0, y0]
    這一步是為了避免一開始w h 是負數，導致x0y0成了右下角座標，x1y1成了左上角座標。
    '''
    boxes1_x0y0x1y1 = tf.concat([tf.minimum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:]),
                                 tf.maximum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:])], axis=-1)
    boxes2_x0y0x1y1 = tf.concat([tf.minimum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:]),
                                 tf.maximum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:])], axis=-1)

    # 兩個矩形的面積
    boxes1_area = (boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) * (
                boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1])
    boxes2_area = (boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) * (
                boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1])

    # 相交矩形的左上角座標、右下角座標，shape 都是 (8, 13, 13, 3, 2)
    left_up = tf.maximum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    right_down = tf.minimum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 相交矩形的面積inter_area。iou
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + K.epsilon())

    # 包圍矩形的左上角座標、右下角座標，shape 都是 (8, 13, 13, 3, 2)
    enclose_left_up = tf.minimum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    enclose_right_down = tf.maximum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 包圍矩形的對角線的平方
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_c2 = K.pow(enclose_wh[..., 0], 2) + K.pow(enclose_wh[..., 1], 2)

    # 兩矩形中心點距離的平方
    p2 = K.pow(boxes1[..., 0] - boxes2[..., 0], 2) + K.pow(boxes1[..., 1] - boxes2[..., 1], 2)

    # 增加av。加上除0保護防止nan。
    atan1 = tf.atan(boxes1[..., 2] / (boxes1[..., 3] + K.epsilon()))
    atan2 = tf.atan(boxes2[..., 2] / (boxes2[..., 3] + K.epsilon()))
    v = 4.0 * K.pow(atan1 - atan2, 2) / (math.pi ** 2)
    a = v / (1 - iou + v)

    ciou = iou - 1.0 * p2 / enclose_c2 - 1.0 * a * v
    ciou = K.expand_dims(ciou, -1)
    return ciou

def bbox_iou(boxes1, boxes2):
    '''
    計算ciou = iou - p2/c2 - av
    :param boxes1: (8, 13, 13, 3, 4)   pred_xywh
    :param boxes2: (8, 13, 13, 3, 4)   label_xywh
    :return:

    舉例時假設pred_xywh和label_xywh的shape都是(1, 4)
    '''

    # 變成左上角座標、右下角座標
    boxes1_x0y0x1y1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_x0y0x1y1 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    '''
    逐個位置比較boxes1_x0y0x1y1[..., :2]和boxes1_x0y0x1y1[..., 2:]，即逐個位置比較[x0, y0]和[x1, y1]，小的留下。
    比如留下了[x0, y0]
    這一步是為了避免一開始w h 是負數，導致x0y0成了右下角座標，x1y1成了左上角座標。
    '''
    boxes1_x0y0x1y1 = tf.concat([tf.minimum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:]),
                                 tf.maximum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:])], axis=-1)
    boxes2_x0y0x1y1 = tf.concat([tf.minimum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:]),
                                 tf.maximum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:])], axis=-1)

    # 兩個矩形的面積
    boxes1_area = (boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) * (
                boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1])
    boxes2_area = (boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) * (
                boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1])

    # 相交矩形的左上角座標、右下角座標，shape 都是 (8, 13, 13, 3, 2)
    left_up = tf.maximum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    right_down = tf.minimum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 相交矩形的面積inter_area。iou
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + K.epsilon())

    return iou

