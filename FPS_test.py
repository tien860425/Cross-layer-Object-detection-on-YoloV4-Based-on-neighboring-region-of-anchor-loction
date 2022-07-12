import os
import time

import numpy as np
from keras import backend as K
from PIL import Image

from utils.utils import letterbox_image
from yolo import YOLO

'''
該FPS測試不包括前處理（歸一化與resize部分）、繪圖。
包括的內容為：網路推理、得分門限篩選、非極大抑制。
使用'img/street.jpg'圖片進行測試，該測試方法參考庫https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

video.py裡面測試的FPS會低於該FPS，因為攝像頭的讀取頻率有限，而且處理過程包含了前處理和繪圖部分。
'''


class FPS_YOLO(YOLO):
    def get_FPS(self, image, test_interval):
        # 調整圖片使其符合輸入要求
        new_image_size = (self.model_image_size[1], self.model_image_size[0])
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        t1 = time.time()
        for _ in range(test_interval):
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time


yolo = FPS_YOLO()
test_interval = 100
img = Image.open('img/street.jpg')
tact_time = yolo.get_FPS(img, test_interval)
print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

