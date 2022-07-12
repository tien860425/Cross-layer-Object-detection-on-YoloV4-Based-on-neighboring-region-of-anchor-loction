# -------------------------------------#
#   調用攝像頭或者視頻進行檢測
#   調用攝像頭直接運行即可
#   調用視頻可以將cv2.VideoCapture()指定路徑
#   視頻的保存並不難，可以百度一下看看
# -------------------------------------#
import time

import cv2
import numpy as np
from keras.layers import Input
from PIL import Image

from yolo import YOLO

yolo = YOLO()
# -------------------------------------#
#   調用攝像頭
#   capture=cv2.VideoCapture("1.mp4")
# -------------------------------------#
capture = cv2.VideoCapture(0)

fps = 0.0
while (True):
    t1 = time.time()
    # 讀取某一幀
    ref, frame = capture.read()
    # 格式轉變，BGRtoRGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 轉變成Image
    frame = Image.fromarray(np.uint8(frame))
    # 進行檢測
    frame = np.array(yolo.detect_image(frame))
    # RGBtoBGR滿足opencv顯示格式
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    fps = (fps + (1. / (time.time() - t1))) / 2
    print("fps= %.2f" % (fps))
    frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video", frame)
    c = cv2.waitKey(1) & 0xff
    if c == 27:
        capture.release()
        break

yolo.close_session()


