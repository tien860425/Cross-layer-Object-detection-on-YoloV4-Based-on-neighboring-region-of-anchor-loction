import numpy as np
import tensorflow as tf
from keras import backend as K

from nets.ious import box_ciou
from nets.yolo4 import yolo_head
from nets.config import NUM_POS, CONF_MASK, LOC_MASK, CLASS_MASK, CONF_WEIGHT, LOC_WEIGHT, CLASS_WEIGHT, IGNORE, DEBUG

# ---------------------------------------------------#
#   平滑標籤
# ---------------------------------------------------#
def _smooth_labels(y_true, label_smoothing):
    num_classes = tf.cast(K.shape(y_true)[-1], dtype=K.floatx())
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes


# ---------------------------------------------------#
#   將預測值的每個特徵層調成真實值
# ---------------------------------------------------#
# def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
#     num_anchors = len(anchors)
#     # ---------------------------------------------------#
#     #   [1, 1, 1, num_anchors, 2]
#     # ---------------------------------------------------#
#     anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
#
#     # ---------------------------------------------------#
#     #   獲得x，y的網格
#     #   (13, 13, 1, 2)
#     # ---------------------------------------------------#
#     grid_shape = K.shape(feats)[1:3]
#     grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
#                     [1, grid_shape[1], 1, 1])
#     grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
#                     [grid_shape[0], 1, 1, 1])
#     grid = K.concatenate([grid_x, grid_y])
#     grid = K.cast(grid, K.dtype(feats))
#
#     # ---------------------------------------------------#
#     #   將預測結果調整成(batch_size,13,13,3,85)
#     #   85可拆分成4 + 1 + 80
#     #   4代表的是中心寬高的調整參數
#     #   1代表的是框的置信度
#     #   80代表的是種類的置信度
#     # ---------------------------------------------------#
#     feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
#
#     # ---------------------------------------------------#
#     #   將預測值調成真實值
#     #   box_xy對應框的中心點
#     #   box_wh對應框的寬和高
#     # ---------------------------------------------------#
#     box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
#     box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
#     box_confidence = K.sigmoid(feats[..., 4:5])
#     box_class_probs = K.sigmoid(feats[..., 5:])
#
#     # ---------------------------------------------------------------------#
#     #   在計算loss的時候返回grid, feats, box_xy, box_wh
#     #   在預測的時候返回box_xy, box_wh, box_confidence, box_class_probs
#     # ---------------------------------------------------------------------#
#     if calc_loss == True:
#         return grid, feats, box_xy, box_wh
#     return box_xy, box_wh, box_confidence, box_class_probs


# ---------------------------------------------------#
#   用於計算每個預測框與真實框的iou
# ---------------------------------------------------#
def box_iou(b1, b2):
    # 13,13,3,1,4
    # 計算左上角的座標和右下角的座標
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # 1,n,4
    # 計算左上角和右下角的座標
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 計算重合面積
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou
def trim_zeros_graph(boxes):
    """
    Often boxes are represented with matrices of shape [N, 4] and are padded with zeros.
    This removes zero boxes.

    Args:
        boxes: [N, 4] matrix of boxes.
        name: name of tensor

    Returns:

    """
    # non_zeros = boxes[:,2:3]*boxes[:,3:4] > 0
    # boxes = tf.boolean_mask(boxes, non_zeros)
    # no_boxes = tf.reduce_sum(tf.cast(non_zeros,tf.int32),axis=-1)

    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros)
    no_boxes = tf.reduce_sum(tf.cast(non_zeros,tf.int32),axis=-1)
    return boxes, no_boxes

def level_select(objindex_list, maximun_boxes,  feature_shapes, y_true_conf_obj,):

    def _compare_probality(args):
        objid = args[0]
        prob_l=[]
        for level_id in range(3):
            fh = feature_shapes[level_id][0]
            fw = feature_shapes[level_id][1]
            fa = tf.reduce_prod(feature_shapes, axis=-1)
            start_idx = tf.reduce_sum(fa[:level_id])
            end_idx = start_idx + fh * fw
            ly=y_true_conf_obj[start_idx:end_idx,...]
            objmask = tf.where(tf.equal(ly[..., 1:2],tf.cast(objid,dtype=tf.float32)),
                                        tf.ones(tf.shape(ly[..., 1:2])),
                                        tf.zeros(tf.shape(ly[..., 1:2])))

            objPosAnchors = tf.maximum(K.sum(objmask),1)
            prob=tf.cond(tf.equal(K.sum(objmask), 0),
                    true_fn=lambda: 1e-10,
                    false_fn=lambda: K.sum(objmask * ly[...,0:1]) / objPosAnchors)
            # prob = K.sum(objmask * ly[...,0:1]) / objPosAnchors
            # prob = tf.Print(prob, [K.sum(ly[...,0:1]),prob,objPosAnchors], message='MidValue: ') #, K.shape(list_batch_gt_conf[l]),K.sum(ignore_mask)
            # prob = tf.Print(prob, [K.sum(ly[..., 0:1]), prob, objPosAnchors], message='IniValue: ')

            prob_l.append(prob)
        lyr =tf.argmax(prob_l)
        return lyr
    layer_per_image = tf.map_fn(
        _compare_probality,
        elems=[objindex_list],
        dtype=tf.int64,)
    padding_gt_box_levels = tf.ones((maximun_boxes - tf.size(objindex_list)), dtype=tf.int64) * -1
    layer_per_image=tf.concat([layer_per_image, padding_gt_box_levels], axis=0)
    # layer_per_image = tf.Print(layer_per_image, [layer_per_image], message='MidValue: ')
    return layer_per_image

def one_batch_confidence_true(gt_box_levels, y_true_conf_obj,num_layer, feature_shapes):
    all_layer=[]
    for level_id in range(num_layer):
        fh = feature_shapes[level_id][0]
        fw = feature_shapes[level_id][1]
        fa = tf.reduce_prod(feature_shapes, axis=-1)
        start_idx = tf.reduce_sum(fa[:level_id])
        end_idx = start_idx + fh * fw
        level_gt_box_indices = tf.where(tf.equal(gt_box_levels, level_id))
        level_gt_box_indices=tf.reshape(level_gt_box_indices,(-1,)) +1
        obj_num =tf.size(level_gt_box_indices)
        y_true_prob_obj=y_true_conf_obj[start_idx:end_idx,:, 0:1]

        def do_level_has_gt_boxes():

            def do_confidence_mark(args):
                objid = args[0]
                y_true_outer_mask=tf.where(
                    tf.equal(y_true_conf_obj[start_idx:end_idx, :,2:3], tf.cast(objid,dtype=tf.float32)),
                                          tf.ones(tf.shape(y_true_conf_obj[start_idx:end_idx, :,1:2])),
                                          tf.zeros(tf.shape(y_true_conf_obj[start_idx:end_idx, :,1:2])))
                y_true_conf_mask=tf.where(
                    tf.equal(y_true_conf_obj[start_idx:end_idx, :,1:2], tf.cast(objid,dtype=tf.float32)),
                                          tf.ones(tf.shape(y_true_conf_obj[start_idx:end_idx, :,1:2])),
                                          tf.zeros(tf.shape(y_true_conf_obj[start_idx:end_idx, :,1:2])))
                # y_true_conf_bool=y_true_conf_obj[start_idx:end_idx, :,-1] == objid

                # y_true_conf_mask_bool = K.cast(K.expand_dims(y_true_conf_mask, axis=-1), 'bool')

                # one_objmask = tf.where(tf.equal(yy,1.0),tf.zeros(tf.shape(yy)),
                #                        tf.ones(tf.shape(yy))
                #                        )
                # one_objmask = tf.Print(one_objmask, [K.sum((1-one_objmask)),objid], message='Add Positive Before: ')
                #
                # xx = y_true_conf_mask
                # xx = tf.Print(xx, [K.sum(xx), objid], message='Add Positive Mid: ')
                objval = y_true_conf_mask * y_true_prob_obj  # mask * probability   y_true_conf_mask:該物件有效區 value=1
                tempobjval=tf.identity(objval)
                # y_true_conf_mask1=tf.reshape(y_true_conf_mask, (fh,fw,3,-1))
                maxval = tf.reduce_max(tempobjval)
                # maxpos_bool = objval >= maxval
                if DEBUG:
                    iouu = y_true_conf_obj[start_idx:end_idx, :, 3]
                    cprob= y_true_conf_obj[start_idx:end_idx, :, 4]
                    confprob = y_true_conf_obj[start_idx:end_idx, :, 5]
                    inx = tf.where(tf.equal(objval, maxval))
                    objval = tf.Print(objval, [inx, objid, maxval,iouu[inx[0,0],inx[0,1]], cprob[inx[0,0],inx[0,1]], confprob[inx[0,0],inx[0,1]]], message='IN:({}) '.format(level_id))
                #y_true_conf_obj[inx[0],inx[1],inx[2],3]
                for i in range(NUM_POS):
                    maxval = tf.cond(
                        tf.greater( tf.reduce_max(tempobjval), 0),
                        lambda:tf.reduce_max(tempobjval),
                        lambda:maxval)
                    # maxval = tf.reduce_max(tempobjval)
                    maxpos = tf.where(tf.greater_equal(tempobjval, maxval),tf.ones(tf.shape(y_true_conf_mask)),
                                       tf.zeros(tf.shape(y_true_conf_mask))
                                       )
                    tempobjval=tempobjval*(1-maxpos)
                maxpos = tf.where(tf.greater_equal(objval, maxval), tf.ones(tf.shape(y_true_conf_mask)),
                                  tf.zeros(tf.shape(y_true_conf_mask))
                                  )   #前k個最大為1，其餘有效區為0
                maxpos=maxpos*objval  # add in 20210811
                zz = y_true_outer_mask+y_true_conf_mask  + maxpos   #k positive sample >1, ，其餘有效區為1
                # onenum = tf.where(tf.equal(zz, 1.0), tf.ones(tf.shape(zz)),
                #                   tf.zeros(tf.shape(zz))
                #                   )
                # zz = tf.Print(zz, [K.sum(maxpos),K.sum(zz)], message='OUT: ')
                return zz

            out_puts=tf.map_fn(
                do_confidence_mark,
                elems=[level_gt_box_indices],
                dtype=tf.float32,)
            # rsize = out_puts.get_shape().as_list()[0]
            # list_level_gt_target=[]
            # for i in range (100):
            #     list_level_gt_target.append(out_puts[i])
            # level_gt_target = tf.concat(list_level_gt_target, axis=-1)
            combind_obj =tf.zeros(tf.shape(y_true_conf_obj[start_idx:end_idx, :, 0:1]))
            def loop_body(b, combind_obj):
                combind_obj=tf.concat([combind_obj,out_puts[b]],axis=-1)
                return b + 1, combind_obj

            _, combind_obj = K.control_flow_ops.while_loop(lambda b, *args: b < obj_num, loop_body, [0, combind_obj])
            combind_obj = tf.reduce_max(combind_obj,axis=-1,keepdims=True)
            # combind_obj = tf.expand_dims(combind_obj,axis=-1)
            # level_gt_target=combind_obj*(-1.0)
            # level_gt_target= tf.where(tf.equal(level_gt_target,-2.0),tf.ones(tf.shape(level_gt_target)),
            #                               level_gt_target)
            level_gt_target=combind_obj
            # one_objmask = tf.where(tf.equal(level_gt_target,1.0),tf.ones(tf.shape(level_gt_target)),
            #                        tf.zeros(tf.shape(level_gt_target)))
            #
            # level_gt_target = tf.Print(level_gt_target, [K.sum(level_gt_target),K.sum(one_objmask),tf.shape(level_gt_target),obj_num], message='Add Positive level {}: '.format(level_id))

            return level_gt_target

        def do_level_has_no_gt_boxes():
            return tf.zeros(tf.shape(y_true_conf_obj[start_idx:end_idx,:, 1:2]))
        level_confidence_target = tf.cond(
            tf.equal(tf.size(level_gt_box_indices), 0),
            do_level_has_no_gt_boxes,
            do_level_has_gt_boxes)
        all_layer.append(level_confidence_target)
    one_batch_conf=tf.concat(all_layer, axis=0)
    return one_batch_conf


# ---------------------------------------------------#
#   loss值計算
# ---------------------------------------------------#
def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, label_smoothing=0.1, print_loss=True, normalize=True):
    # 一共有三層
    num_layers = len(anchors) // 3

    # ---------------------------------------------------------------------------------------------------#
    #   將預測結果和實際ground truth分開，args是[*model_body.output, *y_true]
    #   y_true是一個清單，包含三個特徵層，shape分別為(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    #   yolo_outputs是一個清單，包含三個特徵層，shape分別為(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    # ---------------------------------------------------------------------------------------------------#
    y_true = args[num_layers:2 * num_layers]
    yolo_outputs = args[:num_layers]
    batch_y_true_box = args[2 * num_layers]

    # -----------------------------------------------------------#
    #   13x13的特徵層對應的anchor是[142, 110], [192, 243], [459, 401]
    #   26x26的特徵層對應的anchor是[36, 75], [76, 55], [72, 146]
    #   52x52的特徵層對應的anchor是[12, 16], [19, 36], [40, 28]
    # -----------------------------------------------------------#
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    # 得到input_shpae為416,416
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))

    loss = 0
    num_pos = 0
    # -----------------------------------------------------------#
    #   取出每一張圖片
    #   m的值就是batch_size
    # -----------------------------------------------------------#
    m = K.shape(yolo_outputs[0])[0]
    mf = K.cast(m, K.dtype(yolo_outputs[0]))
    y_true_temp=[]
    y_true_conf_obj=[]
    batch_raw_true_box=[]
    for l in range(num_layers):
        # y_true_l = []
        # -----------------------------------------------------------#
        #   取出其對應的種類(m,13,13,3,80)
        # -----------------------------------------------------------#
        true_class_probs = y_true[l][..., 6:]
        if label_smoothing:
            true_class_probs = _smooth_labels(true_class_probs, label_smoothing)

        # -----------------------------------------------------------#
        #   將yolo_outputs的特徵層輸出進行處理、獲得四個返回值
        #   其中：
        #   grid        (13,13,1,2) 網格座標
        #   raw_pred    (m,13,13,3,85) 尚未處理的預測結果
        #   pred_xy     (m,13,13,3,2) 解碼後的中心座標
        #   pred_wh     (m,13,13,3,2) 解碼後的寬高座標
        # -----------------------------------------------------------#
        # feats, anchors, num_classes, input_shape, calc_loss = False)
        grid,raw_pred, pre_obj_conf, pred_xy, pred_wh = yolo_head(yolo_outputs[l],anchors[anchor_mask[l]],
                                                      num_classes, input_shape, calc_loss=True)

        # -----------------------------------------------------------#
        #   pred_box是解碼後的預測的box的位置
        #   (m,13,13,3,4)
        # -----------------------------------------------------------#
        pred_box = K.concatenate([pred_xy, pred_wh])

        # #-----------------------------------------------------------#
        # #   真實框越大，比重越小，小框的比重更大。
        # #-----------------------------------------------------------#
        box_loss_scale = 2 - (y_true[l][..., 0:1] + y_true[l][..., 2:3]) * (y_true[l][..., 1:2] + y_true[l][..., 3:4])
        #
        # #-----------------------------------------------------------#
        # #   計算Ciou loss
        # #-----------------------------------------------------------#
        raw_true_box_xy_=tf.zeros(tf.shape(y_true[l][..., 0:2]))
        raw_true_box_xy=(raw_true_box_xy_+grid+0.5)/K.cast(K.shape(raw_pred)[1:3][::-1], K.dtype(raw_pred))
        raw_true_box_xy =( raw_true_box_xy-K.concatenate([y_true[l][..., 0:1],y_true[l][..., 1:2]],axis=-1) +
                           raw_true_box_xy + K.concatenate([y_true[l][..., 2:3], y_true[l][..., 3:4]], axis=-1))/2.0
        raw_true_box_w = y_true[l][..., 0:1] + y_true[l][..., 2:3]
        raw_true_box_h = y_true[l][..., 1:2] +y_true[l][..., 3:4]
        raw_true_box_wh =K.concatenate([raw_true_box_w, raw_true_box_h], axis=-1)
        raw_true_box = K.concatenate([raw_true_box_xy,raw_true_box_wh], axis=-1)
        # raw_true_box = tf.Print(raw_true_box, [raw_true_box[0,4,6,2,0:4],pred_box[0,4,6,2,0:4]],
        #                      message='(x,y,w,h): ({}) '.format(l))  # , K.shape(list_batch_gt_conf[l]),K.sum(ignore_mask)
        ciou = box_ciou(pred_box, raw_true_box)
        # minusciou=tf.where(tf.less(ciou,0.0),tf.ones(tf.shape(ciou)),tf.zeros(tf.shape(ciou)))
        # minusciou = tf.Print(minusciou, [K.sum(minusciou)], message='MidValue: ') #, K.shape(list_batch_gt_conf[l]),K.sum(ignore_mask)
        ciou_loss = box_loss_scale * (1 - ciou)
        # class_loss = K.sum(K.binary_crossentropy(true_class_probs, raw_pred[..., 7:], from_logits=True),axis=-1 , keepdims=True)
        class_loss = K.sum(focal_class(true_class_probs, K.sigmoid(raw_pred[..., 5:])),axis=-1 , keepdims=True)

        prob_better= K.exp(-1.0*(1-ciou)) * K.exp(-1.0*class_loss)
        # prob_better = tf.Print(prob_better, [ciou[0,4,6,2,0],class_loss[0,4,6,2,0],prob_better[0,4,6,2,0]],
        #                      message='CIOU, Class, Prob: ({}) '.format(l))  # , K.shape(list_batch_gt_conf[l]),K.sum(ignore_mask)
        y_true_l=tf.concat([ciou_loss,class_loss,pre_obj_conf,pred_box],axis=-1)
        if DEBUG:
            class_prob=K.max(K.sigmoid(raw_pred[..., 5:]), axis=-1)
            class_prob=K.expand_dims(class_prob,-1)
            y_true_l2 = tf.concat([prob_better, y_true[l][..., 4:5], y_true[l][..., 5:6], ciou,class_prob,K.sigmoid(raw_pred[..., 4:5])], axis=-1)
            y_true_conf_obj.append(tf.reshape(y_true_l2,[m, -1, 3, 6]))
        else:
            y_true_l2 = tf.concat([prob_better, y_true[l][..., 4:5], y_true[l][..., 5:6]], axis=-1)
            y_true_conf_obj.append(tf.reshape(y_true_l2,[m, -1, 3, 3]))

        y_true_temp.append(y_true_l)
        batch_raw_true_box.append(raw_true_box)
    batch_y_true_conf_obj = tf.concat(y_true_conf_obj, axis=1)

    #     class_loss=K.sum(class_loss)
    #一張一張圖處理
    #每張圖有正樣本的pixel branch=>obj_mask_bool
    #每張圖有多少標註的物件(object)=>maxobj
    #接著每一物件找出最適合預測的的layer
        # class_loss = tf.Print(class_loss, [class_loss,K.sum(ciou_loss)], message='Print: ')
    # y_true_conf = [[],[],[]]

    feature_shapes=tf.constant([
        [y_true[0].get_shape().as_list()[1], y_true[0].get_shape().as_list()[2]],
        [y_true[1].get_shape().as_list()[1], y_true[1].get_shape().as_list()[2]],
        [y_true[2].get_shape().as_list()[1], y_true[2].get_shape().as_list()[2]]
      ])
    # feature_shapes = tf.constant([[19,19],[38,38],[76,76]])

#
#  每一張圖
#    每一層
#    每個物件在該層的機率(or loss)
#三層stack(axis=-1
#取最大值(axis=-1)
    # batch_boxes, batch_num_boxes=trim_zeros_graph(y_true_box)

    def _level_select(args):
        # objindex = args[0]
        y_true_box = args[0]
        y_true_conf_obj_ = args[1]
        _, num_boxes = trim_zeros_graph(y_true_box)
        maximun_boxes = tf.shape(y_true_box)[0]
        objindex_list=tf.range(num_boxes)+1
        return level_select(
            objindex_list,maximun_boxes, feature_shapes,
            y_true_conf_obj_,
        )
#每張圖每各物件在那層做預測
#[b,100]
    batch_gt_box_levels = tf.map_fn(
        _level_select,
        elems=[batch_y_true_box, batch_y_true_conf_obj],
        dtype=tf.int64,
    )

    def _build_target_confidence(args):
        gt_box_levels = args[0]
        y_true_conf_obj = args[1]
        return one_batch_confidence_true(
            gt_box_levels, y_true_conf_obj,num_layers,feature_shapes
        )

    batch_gt_conf = tf.map_fn(
        _build_target_confidence,
        elems=[batch_gt_box_levels,batch_y_true_conf_obj],
        dtype=tf.float32,
    )
    batch_gt_conf=tf.concat(batch_gt_conf, axis=0)
    # w=tf.shape(batch_gt_conf)[0]
    # list_batch_gt_conf=[]
    # for level_id in range(num_layers):
    #     fh = feature_shapes[level_id][0]
    #     fw = feature_shapes[level_id][1]
    #     fa = tf.reduce_prod(feature_shapes, axis=-1)
    #     start_idx = tf.reduce_sum(fa[:level_id])
    #     end_idx = start_idx + fh * fw
    #     list_batch_gt_conf.append(tf.reshape(batch_gt_conf[:,start_idx:end_idx,:,1],
    #                             (m,fh,fw,5,1)))

    loss = 0
    num_pos = 0
    for l in range(3):
        # iii =tf.where(tf.equal(y_true[l][...,-1],1.0),K.ones(tf.shape(y_true[l][...,-1])) ,
        #               K.zeros(tf.shape(y_true[l][...,-1])))

        fh = feature_shapes[l][0]
        fw = feature_shapes[l][1]
        fa = tf.reduce_prod(feature_shapes, axis=-1)
        # fa = tf.Print(fa, [fh,fw],
        #                      message='hxw: ({}) '.format(l))  # , K.shape(list_batch_gt_conf[l]),K.sum(ignore_mask)
        start_idx = tf.reduce_sum(fa[:l])
        end_idx = start_idx + fh * fw
        l_conf_true=batch_gt_conf[0:,start_idx:end_idx,0:,0:]
        l_conf_true = K.reshape(l_conf_true,(m,fh,fw,3,1))
        true_object_mask_bool = l_conf_true[...,0:] > 2.0
        true_object_mask = K.cast(true_object_mask_bool, 'float32')
        object_mask_bool = l_conf_true[...,0:] > 1.0
        object_mask = K.cast(object_mask_bool, 'float32')
        outer_mask_bool = l_conf_true[...,0:] > 0.0
        outer_mask = K.cast(outer_mask_bool, 'float32')

        #**************************************
        if IGNORE == 0:
            ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
            # object_mask_bool = K.cast(object_mask, 'bool')

            # -----------------------------------------------------------#
            #   對每一張圖片計算ignore_mask
            # -----------------------------------------------------------#
            def loop_body(b, ignore_mask):
                # -----------------------------------------------------------#
                #   取出n個真實框：n,4
                # -----------------------------------------------------------#
                true_box = tf.boolean_mask(batch_raw_true_box[l][b,...,0:4], true_object_mask_bool[b, ..., 0])
                # -----------------------------------------------------------#
                #   計算預測框與真實框的iou
                #   pred_box    13,13,3,4 預測框的座標
                #   true_box    n,4 真實框的座標
                #   iou         13,13,3,n 預測框和真實框的iou
                # -----------------------------------------------------------#
                iou = box_iou(y_true_temp[l][b,...,3:], true_box)

                # -----------------------------------------------------------#
                #   best_iou    13,13,3 每個特徵點與真實框的最大重合程度
                # -----------------------------------------------------------#
                best_iou = K.max(iou, axis=-1)

                # -----------------------------------------------------------#
                #   判斷預測框和真實框的最大iou小於ignore_thresh
                #   則認為該預測框沒有與之對應的真實框
                #   該操作的目的是：
                #   忽略預測結果與真實框非常對應特徵點，因為這些框已經比較准了
                #   不適合當作負樣本，所以忽略掉。
                # -----------------------------------------------------------#
                ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
                return b + 1, ignore_mask

            # -----------------------------------------------------------#
            #   在這個地方進行一個迴圈、迴圈是對每一張圖片進行的
            # -----------------------------------------------------------#
            _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])

            # -----------------------------------------------------------#
            #   ignore_mask用於提取出作為負樣本的特徵點
            #   (m,13,13,3)
            # -----------------------------------------------------------#
            ignore_mask = ignore_mask.stack()
            #   (m,13,13,3,1)
            ignore_mask = K.expand_dims(ignore_mask, -1)

        #**************************************
        if IGNORE==1:

            ignore_mask = tf.where(tf.equal(y_true[l][..., 4:5],0.0), tf.ones(tf.shape(y_true[l][..., 4:5])),
                                   tf.zeros(tf.shape(y_true[l][..., 4:5])))
        if IGNORE==2:
            ignore_mask = tf.where(tf.equal(y_true[l][..., 5:6],0.0), tf.ones(tf.shape(y_true[l][..., 5:6])),
                                   tf.zeros(tf.shape(y_true[l][..., 5:6])))

        all_mask=[true_object_mask,object_mask,outer_mask]
        conf_mask=all_mask[CONF_MASK]
        loc_mask=all_mask[LOC_MASK]
        class_mask=all_mask[CLASS_MASK]
        conf_weight=1
        if CONF_WEIGHT==100:
            conf_weight=(l_conf_true[...,0:]-1)
        else:
            conf_weight = CONF_WEIGHT

        # confidence_loss = (l_conf_true[...,0:]-1)*true_object_mask * K.binary_crossentropy(true_object_mask, y_true_temp[l][...,2:3], from_logits=False)+ \
        #     (1-true_object_mask) * K.binary_crossentropy(true_object_mask, y_true_temp[l][...,2:3], from_logits=False) * ignore_mask
        confidence_loss = conf_weight*conf_mask * K.binary_crossentropy(conf_mask, y_true_temp[l][...,2:3], from_logits=False)+ \
            (1-conf_mask) * K.binary_crossentropy(conf_mask, y_true_temp[l][...,2:3], from_logits=False) * ignore_mask
        #
        class_loss = CLASS_WEIGHT *class_mask *y_true_temp[l][...,1:2]
        location_loss =LOC_WEIGHT * loc_mask *y_true_temp[l][...,0:1]
        location_loss = K.sum(tf.where(tf.is_nan(location_loss), tf.zeros_like(location_loss), location_loss))
        confidence_loss = K.sum(tf.where(tf.is_nan(confidence_loss), tf.zeros_like(confidence_loss), confidence_loss))
        class_loss = K.sum(tf.where(tf.is_nan(class_loss), tf.zeros_like(class_loss), class_loss))

        # -----------------------------------------------------------#
        #   計算正樣本數量
        # -----------------------------------------------------------#
        num_true_pos = tf.maximum(K.sum(true_object_mask), 1)
        num_pos = tf.maximum(K.sum(object_mask), 1)
        num_out = tf.maximum(K.sum(outer_mask), 1)
        normal_size=[num_true_pos,num_pos,num_out]
        if normalize:
            location_loss= location_loss / normal_size[LOC_MASK  ]
            confidence_loss = confidence_loss/ normal_size[CONF_MASK ]
            class_loss = class_loss/ normal_size[CLASS_MASK  ]

        loss += location_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, location_loss, confidence_loss,
                         class_loss,K.sum(true_object_mask),K.sum(object_mask), K.sum(ignore_mask)], message='loss:{} '.format(l)) #, K.shape(list_batch_gt_conf[l]),K.sum(ignore_mask)

    # K.shape(list_batch_gt_conf[l])[0], K.shape(list_batch_gt_conf[l])[1],
    # K.shape(list_batch_gt_conf[l])[2]
    # if normalize:
    #     loss = loss / num_pos
    # else:
    #     loss = loss / mf
    return loss

def focal_class(y_true, y_pred, alpha=0.25, gamma=2.0):

    labels = y_true[..., :]
    alpha_factor = K.ones_like(labels) * alpha
    alpha_factor = tf.where(K.equal(labels, 1), alpha_factor, 1 - alpha_factor)
    # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
    focal_weight = tf.where(K.equal(labels, 1), 1 - y_pred, y_pred)
    focal_weight = alpha_factor * focal_weight ** gamma
    cls_loss = focal_weight * K.binary_crossentropy(labels, y_pred)

    return cls_loss
