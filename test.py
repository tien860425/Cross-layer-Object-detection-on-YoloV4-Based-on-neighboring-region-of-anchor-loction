#--------------------------------------------#
#   該部分代碼只用於看網路結構，並非測試代碼
#   map測試請看get_dr_txt.py、get_gt_txt.py
#   和get_map.py
#--------------------------------------------#
from keras.layers import Input

from nets.yolo4 import yolo_body

if __name__ == "__main__":
    inputs = Input([416, 416, 3])
    model = yolo_body(inputs, 3, 20)
    model.summary()

    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)

