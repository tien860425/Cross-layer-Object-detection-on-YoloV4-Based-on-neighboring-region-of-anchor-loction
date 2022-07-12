# ----------------------------------------------------#
#   獲取測試集的detection-result和images-optional
#   具體視頻教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
# ----------------------------------------------------#
from yolo import YOLO
from PIL import Image
from keras.layers import Input
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
from keras.models import load_model
from utils.utils import letterbox_image
from nets.yolo4 import yolo_body, yolo_eval
from tqdm import tqdm
import colorsys
import numpy as np
import os
import json

class mAP_YOLO(YOLO):
    # ---------------------------------------------------#
    #   獲得所有的分類
    # ---------------------------------------------------#
    def generate(self):
        self.score = 0.01
        self.iou = 0.5
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # 計算anchor數量
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # 載入模型，如果原來的模型裡已經包括了模型結構則直接載入。
        # 否則先構建模型再載入
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

        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           num_classes, self.input_image_shape, max_boxes=self.max_boxes,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    # ---------------------------------------------------#
    #   檢測圖片
    # ---------------------------------------------------#
    def detect_image(self, image_id, image):
        global detectresultjson
        f = open("./input_coco/detection-results/" + image_id + ".txt", "w")
        # 調整圖片使其符合輸入要求
        boxed_image = letterbox_image(image, self.model_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # 預測結果
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        for i, c in enumerate(out_classes):
            predicted_class = self.class_names[int(c)]
            score = str(out_scores[i])

            top, left, bottom, right = out_boxes[i]
            f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

            cat = int(c)
            if cat >= 0 and cat <= 10:
                cat = cat + 1
            elif cat >= 11 and cat <= 23:
                cat = cat + 2
            elif cat >= 24 and cat <= 25:
                cat = cat + 3
            elif cat >= 26 and cat <= 39:
                cat = cat + 5
            elif cat >= 40 and cat <= 59:
                cat = cat + 6
            elif cat == 60:
                cat = cat + 7
            elif cat == 61:
                cat = cat + 9
            elif cat >= 62 and cat <= 72:
                cat = cat + 10
            elif cat >= 73 and cat <= 79:
                cat = cat + 11
            # '%s/bbox/%s.json' % (result_dir, im_name.split('.')[0])
            image_id=image_id.lstrip('0')
            detectresultjson=detectresultjson+\
            '{"image_id": %i,"category_id": %i,"bbox": [%.1f, %.1f, %.1f, %.1f], "score": %.3f},' \
             %(int(image_id), cat,left, top, right-left, bottom-top,out_scores[i].item() )
            # d["image_id"]= int(image_id)
            # d["category_id"]=cat
            # d["bbox"]=[round((left+right)/2,1),round((top+bottom)/2,1), round(right-left,1), round(bottom-top,1)]
            # d["score"]=out_scores[i].item()
            # detectresultjson.append(d)
        f.close()
        return


yolo = mAP_YOLO()

image_path = '../dataset/mscocoval.txt'
detectresultjson='['
if not os.path.exists("./input_coco"):
    os.makedirs("./input_coco")
if not os.path.exists("./input_coco/detection-results"):
    os.makedirs("./input_coco/detection-results")
if not os.path.exists("./input_coco/images-optional"):
    os.makedirs("./input_coco/images-optional")
with open(image_path) as f:
    lines = f.readlines()
for line in tqdm(lines):
    image_path = line.split()[0]
    image = Image.open(image_path)
    image_id =  line.split()[0].split('/')[-1].split('.')[0]
    # 開啟後在之後計算mAP可以視覺化
    # image.save("./input/images-optional/"+image_id+".jpg")
    yolo.detect_image(image_id, image)
detectresultjson=detectresultjson[0:-1]+"]"
with open('detect_result.json', 'w') as f:
    f.write(detectresultjson)
    # json.dump(detectresultjson, f)
print("Conversion completed!")

