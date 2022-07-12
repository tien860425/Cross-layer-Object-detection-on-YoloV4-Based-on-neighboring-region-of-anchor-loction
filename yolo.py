import colorsys
import copy
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

from nets.yolo4 import yolo_body, yolo_eval, yolo_eval_3
from utils.utils import letterbox_image


# --------------------------------------------#
#   使用自己訓練好的模型預測需要修改2個參數
#   model_path和classes_path都需要修改！
#   如果出現shape不匹配，一定要注意
#   訓練時的model_path和classes_path參數的修改
# --------------------------------------------#
class YOLO(object):
    _defaults = {
        "model_path": 'model_data/trained_weights_stage_1.h5',
        "anchors_path": 'model_data/yolo_anchors_416.txt',
        "classes_path": 'model_data/voc_classes.txt',
        "score": 0.5,
        "iou": 0.3,
        "max_boxes": 100,
        # 顯存比較小可以使用416x416
        # 顯存比較大可以使用608x608
        "model_image_size": (416, 416),
        # ---------------------------------------------------------------------#
        #   該變數用於控制是否使用letterbox_image對輸入圖像進行不失真的resize，
        #   在多次測試後，發現關閉letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image": False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化yolo
    # ---------------------------------------------------#
    def __init__(self, show=0, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()

        if show==0:
            self.boxes, self.scores, self.classes = self.generate()
        if show==3:
            self.boxes_3, self.scores_3 = self.generate_3()

    # ---------------------------------------------------#
    #   獲得所有的分類
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   獲得所有的先驗框
    # ---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    # ---------------------------------------------------#
    #   載入模型
    # ---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # ---------------------------------------------------#
        #   計算先驗框的數量和種類的數量
        # ---------------------------------------------------#
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # ---------------------------------------------------------#
        #   載入模型，如果原來的模型裡已經包括了模型結構則直接載入。
        #   否則先構建模型再載入
        # ---------------------------------------------------------#
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 畫框設置不同的顏色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # 打亂顏色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2,))

        # ---------------------------------------------------------#
        #   在yolo_eval函數中，我們會對預測結果進行後處理
        #   後處理的內容包括，解碼、非極大抑制、門限篩選等
        # ---------------------------------------------------------#
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           num_classes, self.input_image_shape, max_boxes=self.max_boxes,
                                           score_threshold=self.score, iou_threshold=self.iou,
                                           letterbox_image=self.letterbox_image)
        return boxes, scores, classes

    # ---------------------------------------------------#
    #   檢測圖片
    # ---------------------------------------------------#
    def detect_image(self, image):
        # ---------------------------------------------------------#
        #   給圖像增加灰條，實現不失真的resize
        #   也可以直接resize進行識別
        # ---------------------------------------------------------#
        if self.letterbox_image:
            boxed_image = letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
        else:
            boxed_image = image.convert('RGB')
            boxed_image = boxed_image.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        # ---------------------------------------------------------#
        #   添加上batch_size維度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(image_data, 0)

        # ---------------------------------------------------------#
        #   將圖像輸入網路當中進行預測！
        # ---------------------------------------------------------#
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0})

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # ---------------------------------------------------------#
        #   設置字體
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='font/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        thickness = max((image.size[0] + image.size[1]) // 300, 1)

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # 畫框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def close_session(self):
        self.sess.close()

    # ---------------------------------------------------#
    #   載入模型
    # ---------------------------------------------------#
    def generate_3(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # ---------------------------------------------------#
        #   計算先驗框的數量和種類的數量
        # ---------------------------------------------------#
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # ---------------------------------------------------------#
        #   載入模型，如果原來的模型裡已經包括了模型結構則直接載入。
        #   否則先構建模型再載入
        # ---------------------------------------------------------#
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None, None, 3)), 3, num_classes)
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 畫框設置不同的顏色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # 打亂顏色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2,))
        # ---------------------------------------------------------#
        #   在yolo_eval函數中，我們會對預測結果進行後處理
        #   後處理的內容包括，解碼、非極大抑制、門限篩選等
        # ---------------------------------------------------------#
        boxes, scores = yolo_eval_3(self.yolo_model.output,self.anchors,
                                           num_classes,
                                             self.input_image_shape, max_boxes=self.max_boxes,
                                           score_threshold=self.score, iou_threshold=self.iou,
                                           letterbox_image=self.letterbox_image)
        return boxes, scores

    # ---------------------------------------------------#
    #   檢測圖片
    # ---------------------------------------------------#

    def detect_image_3(self, image,y_true):
        start = timer()
        self.y_true=y_true
        # ---------------------------------------------------------#
        #   給圖像增加灰條，實現不失真的resize
        #   也可以直接resize進行識別
        # ---------------------------------------------------------#
        if self.letterbox_image:
            boxed_image = letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
        else:
            boxed_image = image.convert('RGB')
            boxed_image = boxed_image.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        # ---------------------------------------------------------#
        #   添加上batch_size維度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(image_data, 0)


        # 預測結果
        out_boxes, out_scores = self.sess.run(
            [self.boxes_3, self.scores_3],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        imcount=[]
        images = []
        for l in range(3):
            images0=[]
            main_index=np.where(y_true[l][...,-1]==1)
            center_grids=y_true[l][main_index]
            no_objs=center_grids.shape[0]
            for i in range(no_objs):
                img_s=[]
                # add box
                box_score_class = []
                for ii in range(4):
                    box_score_class.append(out_boxes[l][main_index][i][ii])
                box_score_class.append(np.max(out_scores[l][main_index][i][:]))
                box_score_class.append(np.argmax(out_scores[l][main_index][i][:]))
                box_score_class.append(main_index[1][i])  #grid-x axis
                box_score_class.append(main_index[2][i])  #grid-y axis
                img_s.append(box_score_class)

                objnum=y_true[l][main_index][i][...,4]
                sub_index = np.where(y_true[l][..., 4] == objnum)
                sub_grids = y_true[l][sub_index]
                no_eff = sub_grids.shape[0]
                for iii in range(no_eff):
                    if np.array_equal(center_grids[i],sub_grids[iii]):
                        continue
                    box_score_class_ = []
                    for aa in range(4):
                        box_score_class_.append(out_boxes[l][sub_index][iii][aa])
                    box_score_class_.append(np.max(out_scores[l][sub_index][iii][:]))
                    box_score_class_.append(np.argmax(out_scores[l][sub_index][iii][:]))
                    box_score_class_.append(sub_index[1][iii])  # grid-x axis
                    box_score_class_.append(sub_index[2][iii])  # grid-y axis

                    img_s.append(box_score_class_)
                images0.append(img_s)
            images.append(images0)
            print('Layer {} Found {} boxes for {}'.format(l,no_objs, 'img'))
        imcount.append(no_objs)
        # 設置字體
        font = ImageFont.truetype(font='font/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        Images=[]
        for i in range(3):
            Image_l=[]
            # if len(Images[i])==0:
            for j in range(len(images[i])): # number of objects in this layer
                Image_o=[]
                for k in range(len(images[i][j])):  # predict in effective area

                    image_=image.copy()
                    c=images[i][j][k][5]
                    predicted_class = self.class_names[c]
                    box = images[i][j][k][0:4]
                    score = images[i][j][k][4]
                    grid_x = images[i][j][k][6]
                    grid_y = images[i][j][k][7]
                    top, left, bottom, right = box
                    top = top - 5
                    left = left - 5
                    bottom = bottom + 5
                    right = right + 5

                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                    # 畫框框
                    if k==0:
                        label = '***{} {:.2f}'.format(predicted_class, score)
                        print('Center grid:x={}, y={}'.format(grid_x, grid_y))
                    else:
                        label = '{} {:.2f}'.format(predicted_class, score)
                        print('grid:x={}, y={}'.format(grid_x,grid_y))

                    draw = ImageDraw.Draw(image_)
                    label_size = draw.textsize(label, font)
                    label = label.encode('utf-8')

                    print('box:({},{},{},{}), score:{:.4f}, class:{}_{}/{}/{} '.format(
                        left, top, right,bottom, score, predicted_class,i+1,j+1,k+1
                    ))

                    if top - label_size[1] >= 0:
                        text_origin = np.array([left, top - label_size[1]])
                    else:
                        text_origin = np.array([left, top + 1])

                    for ix in range(thickness):
                        draw.rectangle(
                            [left + ix, top + ix, right - ix, bottom - i],
                            outline=self.colors[c])
                    draw.rectangle(
                        [tuple(text_origin), tuple(text_origin + label_size)],
                        fill=self.colors[c])
                    draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
                    del draw
                # image_.show()

                    Image_o.append(image_)
                Image_l.append(Image_o)
            Images.append(Image_l)

        end = timer()
        print(end - start)
        return Images, imcount
