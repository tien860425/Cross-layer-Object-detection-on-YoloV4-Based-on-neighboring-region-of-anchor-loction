from functools import wraps

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import (Add, Concatenate, Conv2D, MaxPooling2D, UpSampling2D,
                          ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from utils.utils import compose

from nets.CSPdarknet53 import darknet_body


# --------------------------------------------------#
#   單次卷積DarknetConv2D
#   如果步長為2則自己設定padding方式。
#   測試中發現沒有l2正則化效果更好，所以去掉了l2正則化
# --------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    # darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs = {}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


# ---------------------------------------------------#
#   卷積塊 -> 卷積 + 標準化 + 啟動函數
#   DarknetConv2D + BatchNormalization + LeakyReLU
# ---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


# ---------------------------------------------------#
#   進行五次卷積
# ---------------------------------------------------#
def make_five_convs(x, num_filters):
    # 五次卷積
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    return x


# ---------------------------------------------------#
#   Panet網路的構建，並且獲得預測結果
# ---------------------------------------------------#
def yolo_body(inputs, num_anchors, num_classes):
    # ---------------------------------------------------#
    #   生成CSPdarknet53的主幹模型
    #   獲得三個有效特徵層，他們的shape分別是：
    #   52,52,256
    #   26,26,512
    #   13,13,1024
    # ---------------------------------------------------#
    feat1, feat2, feat3 = darknet_body(inputs)

    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    P5 = DarknetConv2D_BN_Leaky(512, (1, 1))(feat3)
    P5 = DarknetConv2D_BN_Leaky(1024, (3, 3))(P5)
    P5 = DarknetConv2D_BN_Leaky(512, (1, 1))(P5)
    # 使用了SPP結構，即不同尺度的最大池化後堆疊。
    maxpool1 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(P5)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(P5)
    maxpool3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(P5)
    P5 = Concatenate()([maxpool1, maxpool2, maxpool3, P5])
    P5 = DarknetConv2D_BN_Leaky(512, (1, 1))(P5)
    P5 = DarknetConv2D_BN_Leaky(1024, (3, 3))(P5)
    P5 = DarknetConv2D_BN_Leaky(512, (1, 1))(P5)

    # 13,13,512 -> 13,13,256 -> 26,26,256
    P5_upsample = compose(DarknetConv2D_BN_Leaky(256, (1, 1)), UpSampling2D(2))(P5)
    # 26,26,512 -> 26,26,256
    P4 = DarknetConv2D_BN_Leaky(256, (1, 1))(feat2)
    # 26,26,256 + 26,26,256 -> 26,26,512
    P4 = Concatenate()([P4, P5_upsample])

    # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    P4 = make_five_convs(P4, 256)

    # 26,26,256 -> 26,26,128 -> 52,52,128
    P4_upsample = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(P4)
    # 52,52,256 -> 52,52,128
    P3 = DarknetConv2D_BN_Leaky(128, (1, 1))(feat1)
    # 52,52,128 + 52,52,128 -> 52,52,256
    P3 = Concatenate()([P3, P4_upsample])

    # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
    P3 = make_five_convs(P3, 128)

    # ---------------------------------------------------#
    #   第三個特徵層
    #   y3=(batch_size,52,52,3,85)
    # ---------------------------------------------------#
    P3_output = DarknetConv2D_BN_Leaky(256, (3, 3))(P3)
    P3_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1),
                              kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01))(P3_output)

    # 52,52,128 -> 26,26,256
    P3_downsample = ZeroPadding2D(((1, 0), (1, 0)))(P3)
    P3_downsample = DarknetConv2D_BN_Leaky(256, (3, 3), strides=(2, 2))(P3_downsample)
    # 26,26,256 + 26,26,256 -> 26,26,512
    P4 = Concatenate()([P3_downsample, P4])
    # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    P4 = make_five_convs(P4, 256)

    # ---------------------------------------------------#
    #   第二個特徵層
    #   y2=(batch_size,26,26,3,85)
    # ---------------------------------------------------#
    P4_output = DarknetConv2D_BN_Leaky(512, (3, 3))(P4)
    P4_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1),
                              kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01))(P4_output)

    # 26,26,256 -> 13,13,512
    P4_downsample = ZeroPadding2D(((1, 0), (1, 0)))(P4)
    P4_downsample = DarknetConv2D_BN_Leaky(512, (3, 3), strides=(2, 2))(P4_downsample)
    # 13,13,512 + 13,13,512 -> 13,13,1024
    P5 = Concatenate()([P4_downsample, P5])
    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    P5 = make_five_convs(P5, 512)

    # ---------------------------------------------------#
    #   第一個特徵層
    #   y1=(batch_size,13,13,3,85)
    # ---------------------------------------------------#
    P5_output = DarknetConv2D_BN_Leaky(1024, (3, 3))(P5)
    P5_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1),
                              kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01))(P5_output)

    return Model(inputs, [P5_output, P4_output, P3_output])


# ---------------------------------------------------#
#   將預測值的每個特徵層調成真實值
# ---------------------------------------------------#
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    # ---------------------------------------------------#
    #   [1, 1, 1, num_anchors, 2]
    # ---------------------------------------------------#
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # ---------------------------------------------------#
    #   獲得x，y的網格
    #   (13, 13, 1, 2)
    # ---------------------------------------------------#
    grid_shape = K.shape(feats)[1:3]
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    # ---------------------------------------------------#
    #   將預測結果調整成(batch_size,13,13,3,85)
    #   85可拆分成4 + 1 + 80
    #   4代表的是中心寬高的調整參數
    #   1代表的是框的置信度
    #   80代表的是種類的置信度
    # ---------------------------------------------------#
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # ---------------------------------------------------#
    #   將預測值調成真實值
    #   box_xy對應框的中心點
    #   box_wh對應框的寬和高
    # ---------------------------------------------------#
    box_xy_ = tf.zeros(tf.shape(feats[..., 0:2]))
    box_xy=(box_xy_+0.5 + grid)/K.cast(grid_shape[::-1], K.dtype(feats))

    box_l = K.sigmoid(feats[..., 0:1])
    box_t = K.sigmoid(feats[..., 1:2])
    box_r = K.sigmoid(feats[..., 2:3])
    box_b = K.sigmoid(feats[..., 3:4])
    box_xy=(box_xy-tf.concat([box_l,box_t],axis=-1)+box_xy+tf.concat([box_r,box_b],axis=-1))/2.0
    # box_lt =K.concatenate([box_l, box_t], axis=-1)
    # box_rb = K.concatenate([box_r, box_b], axis=-1)
    # box_xy = (box_xy -box_lt + box_xy + box_rb)/2.0
    box_w = box_l + box_r
    box_h =  box_t + box_b
    box_wh = K.concatenate([box_w, box_h], axis=-1)
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # ---------------------------------------------------------------------#
    #   在計算loss的時候返回grid, feats, box_xy, box_wh
    #   在預測的時候返回box_xy, box_wh, box_confidence, box_class_probs
    # ---------------------------------------------------------------------#
    if calc_loss == True:
        return grid, feats,box_confidence, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


# ---------------------------------------------------#
#   對box進行調整，使其符合真實圖片的樣子
# ---------------------------------------------------#
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    # -----------------------------------------------------------------#
    #   把y軸放前面是因為方便預測框和圖像的寬高進行相乘
    # -----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    # -----------------------------------------------------------------#
    #   這裡求出來的offset是圖像有效區域相對于圖像左上角的偏移情況
    #   new_shape指的是寬高縮放情況
    # -----------------------------------------------------------------#
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


# ---------------------------------------------------#
#   獲取每個box和它的得分
# ---------------------------------------------------#
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape, letterbox_image):
    # -----------------------------------------------------------------#
    #   將預測值調成真實值
    #   box_xy : -1,13,13,3,2;
    #   box_wh : -1,13,13,3,2;
    #   box_confidence : -1,13,13,3,1;
    #   box_class_probs : -1,13,13,3,80;
    # -----------------------------------------------------------------#
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    # -----------------------------------------------------------------#
    #   在圖像傳入網路預測前會進行letterbox_image給圖像周圍添加灰條
    #   因此生成的box_xy, box_wh是相對於有灰條的圖像的
    #   我們需要對齊進行修改，去除灰條的部分。
    #   將box_xy、和box_wh調節成y_min,y_max,xmin,xmax
    # -----------------------------------------------------------------#
    if letterbox_image:
        boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    else:
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)

        input_shape = K.cast(input_shape, K.dtype(box_yx))
        image_shape = K.cast(image_shape, K.dtype(box_yx))

        boxes = K.concatenate([
            box_mins[..., 0:1] * image_shape[0],  # y_min
            box_mins[..., 1:2] * image_shape[1],  # x_min
            box_maxes[..., 0:1] * image_shape[0],  # y_max
            box_maxes[..., 1:2] * image_shape[1]  # x_max
        ])
    # -----------------------------------------------------------------#
    #   獲得最終得分和框的位置
    # -----------------------------------------------------------------#
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

def yolo_boxes_and_scores_(feats, anchors, num_classes, input_shape, image_shape, letterbox_image):
    # -----------------------------------------------------------------#
    #   將預測值調成真實值
    #   box_xy : -1,13,13,3,2;
    #   box_wh : -1,13,13,3,2;
    #   box_confidence : -1,13,13,3,1;
    #   box_class_probs : -1,13,13,3,80;
    # -----------------------------------------------------------------#
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    # -----------------------------------------------------------------#
    #   在圖像傳入網路預測前會進行letterbox_image給圖像周圍添加灰條
    #   因此生成的box_xy, box_wh是相對於有灰條的圖像的
    #   我們需要對齊進行修改，去除灰條的部分。
    #   將box_xy、和box_wh調節成y_min,y_max,xmin,xmax
    # -----------------------------------------------------------------#
    if letterbox_image:
        boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    else:
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)

        input_shape = K.cast(input_shape, K.dtype(box_yx))
        image_shape = K.cast(image_shape, K.dtype(box_yx))

        boxes = K.concatenate([
            box_mins[..., 0:1] * image_shape[0],  # y_min
            box_mins[..., 1:2] * image_shape[1],  # x_min
            box_maxes[..., 0:1] * image_shape[0],  # y_max
            box_maxes[..., 1:2] * image_shape[1]  # x_max
        ])
    # -----------------------------------------------------------------#
    #   獲得最終得分和框的位置
    # -----------------------------------------------------------------#
    # boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    # box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores



# ---------------------------------------------------#
#   圖片預測
# ---------------------------------------------------#
def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5,
              letterbox_image=True):
    # ---------------------------------------------------#
    #   獲得特徵層的數量，有效特徵層的數量為3
    # ---------------------------------------------------#
    num_layers = len(yolo_outputs)
    # -----------------------------------------------------------#
    #   13x13的特徵層對應的anchor是[142, 110], [192, 243], [459, 401]
    #   26x26的特徵層對應的anchor是[36, 75], [76, 55], [72, 146]
    #   52x52的特徵層對應的anchor是[12, 16], [19, 36], [40, 28]
    # -----------------------------------------------------------#
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # -----------------------------------------------------------#
    #   這裡獲得的是輸入圖片的大小，一般是416x416
    # -----------------------------------------------------------#
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    # -----------------------------------------------------------#
    #   對每個特徵層進行處理
    # -----------------------------------------------------------#
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape,
                                                    image_shape, letterbox_image)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    # -----------------------------------------------------------#
    #   將每個特徵層的結果進行堆疊
    # -----------------------------------------------------------#
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    # -----------------------------------------------------------#
    #   判斷得分是否大於score_threshold
    # -----------------------------------------------------------#
    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # -----------------------------------------------------------#
        #   取出所有box_scores >= score_threshold的框，和成績
        # -----------------------------------------------------------#
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        # -----------------------------------------------------------#
        #   非極大抑制
        #   保留一定區域內得分最大的框
        # -----------------------------------------------------------#
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        # -----------------------------------------------------------#
        #   獲取非極大抑制後的結果
        #   下列三個分別是
        #   框的位置，得分與種類
        # -----------------------------------------------------------#
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def yolo_eval_3(yolo_outputs,
              anchors,
              num_classes,
              y_true,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5,
              letterbox_image=True):
    # 獲得特徵層的數量
    num_layers = len(yolo_outputs)
    # 特徵層1對應的anchor是678
    # 特徵層2對應的anchor是345
    # 特徵層3對應的anchor是012
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    input_shape = K.shape(yolo_outputs[0])[1:3] * 32

    boxes = []
    scores = []
    classesx =[]
    # 對每個特徵層進行處理
    for l in range(num_layers):
        #[?,13,13,3,[x1,y1,x2,y2])
        _boxes, _box_scores = yolo_boxes_and_scores_(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape,
                                                    image_shape,letterbox_image)
        # boxes.append(_boxes)
        # box_scores.append(_box_scores)
    # 將每個特徵層的結果進行堆疊
    #     boxes = K.concatenate(boxes, axis=0)
    #     box_scores = K.concatenate(box_scores, axis=0)
        main_indices = tf.where(tf.equal(y_true[l][..., -1], 1))
        # gg = tf.gather_nd(y_true[l], tmp_indices)
        def getobjboxes(args):
            idx=args[0]
            image=[]
            imgcontext=[]
            imgcontext.append(tf.gather_nd(_boxes,idx))
            imgcontext.append(tf.gather_nd(_box_scores, idx))
            image.append(imgcontext)
            # centergrid=tf.gather_nd(y_true[l],idx)
            # objno=centergrid[...,4]
            # m_indices = tf.where(tf.equal(y_true[l][..., 4], objno))

            # # def tensorequal(a, b):
            # #     aeqb = tf.equal(a, b)
            # #     aeqb_int = tf.to_int32(aeqb)
            # #     result = tf.equal(tf.reduce_sum(aeqb_int), tf.reduce_sum(tf.ones_like(aeqb_int)))
            # #     return result
            #
            # def getsubmap(idx_i):
            #     imgcontext = []
            #     imgcontext.append(tf.gather_nd(_boxes, idx_i))
            #     imgcontext.append(tf.gather_nd(_box_scores, idx_i))
            #     image.append(imgcontext)
            #
            # # def getobjotherboxes(args):
            # #     idx_i = args[0]
            # #     tf.cond(tf.not_equal(idx_i,idx),getsubmap(idx_i))
            #
            #
            # tf.map_fn(
            #     getsubmap,
            #     elems=[m_indices],
            #     dtype=tf.float32, )
            # tf.concat
            return image

        images_list = tf.map_fn(
            getobjboxes,
            elems=[main_indices],
            dtype=tf.float32, )
        mask = _box_scores >= score_threshold
        max_boxes_tensor = K.constant(max_boxes, dtype='int32')
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(num_classes):
            # 取出所有box_scores >= score_threshold的框，和成績
            class_boxes = tf.boolean_mask(_boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(_box_scores[:, c], mask[:, c])

            # 非極大抑制，去掉box重合程度高的那一些
            nms_index = tf.image.non_max_suppression(
                class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

            # 獲取非極大抑制後的結果
            # 下列三個分別是
            # 框的位置，得分與種類
            class_boxes = K.gather(class_boxes, nms_index)
            class_box_scores = K.gather(class_box_scores, nms_index)
            classes = K.ones_like(class_box_scores, 'int32') * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = K.concatenate(boxes_, axis=0)
        scores_ = K.concatenate(scores_, axis=0)
        classes_ = K.concatenate(classes_, axis=0)
        boxes.append(boxes_)
        scores.append(scores_)
        classesx.append(classes_)
    # boxes = K.concatenate(boxes, axis=0)
    # scores = K.concatenate(scores, axis=0)
    # classesx = K.concatenate(classesx, axis=0)

    return boxes, scores, classesx