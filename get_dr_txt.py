# ----------------------------------------------------#
#   獲取測試集的detection-result和images-optional
#   具體視頻教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
# ----------------------------------------------------#
import colorsys
import os

import numpy as np
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from PIL import Image
from tqdm import tqdm
from keras.models import load_model
from nets.yolo4 import yolo_body, yolo_eval
from utils.utils import letterbox_image
from yolo import YOLO


'''
	這裡設置的門限值較低是因為計算map需要用到不同門限條件下的Recall和Precision值。
	所以只有保留的框足夠多，計算的map才會更精確，詳情可以瞭解map的原理。
	計算map時輸出的Recall和Precision值指的是門限為0.5時的Recall和Precision值。

	此處獲得的./input/detection-results/裡面的txt的框的數量會比直接predict多一些，這是因為這裡的門限低，
	目的是為了計算不同門限條件下的Recall和Precision值，從而實現map的計算。

	這裡的self.iou指的是非極大抑制所用到的iou，具體的可以瞭解非極大抑制的原理，
	如果低分框與高分框的iou大於這裡設定的self.iou，那麼該低分框將會被剔除。

	可能有些人知道有0.5和0.5:0.95的mAP，這裡的self.iou=0.5不代表mAP0.5。
	如果想要設定mAP0.x，比如設定mAP0.75，可以去get_map.py設定MINOVERLAP。
	'''
class mAP_YOLO(YOLO):
    # ---------------------------------------------------#
    #   獲得所有的分類
    # ---------------------------------------------------#
    def generate(self):
        self.score = 0.01
        self.iou = 0.5
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
    def detect_image(self, image_id, image):
        f = open("./input/detection-results/" + image_id + ".txt", "w")
        # ---------------------------------------------------------#
        #   給圖像增加灰條，實現不失真的resize
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

        for i, c in enumerate(out_classes):
            predicted_class = self.class_names[int(c)]
            score = str(out_scores[i])

            top, left, bottom, right = out_boxes[i]
            f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return


yolo = mAP_YOLO()

image_ids = open('../dataset/VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")

for image_id in tqdm(image_ids):
    image_path = "../dataset/VOCdevkit/VOC2007/JPEGImages/" + image_id + ".jpg"
    image = Image.open(image_path)
    # 開啟後在之後計算mAP可以視覺化
    # image.save("./input/images-optional/"+image_id+".jpg")
    yolo.detect_image(image_id, image)

print("Conversion completed!")


