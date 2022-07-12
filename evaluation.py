# _*_  coding utf-8 _*_
# 開發單位：NTTU
# 開發人員： M. Tien
# 開發時間： 2021/1/13下午 01:01
#文件名稱： evaluation.py
# 開發工具： PyCharm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab,json

if __name__ == "__main__":
    cocoGt = COCO('../dataset/mscoco2017/annotations/instances_val2017.json')        #標注文件的路徑及文件名，json文件形式
    cocoDt = cocoGt.loadRes('results_coco/detect_result.json')  #自己的生成的結果的路徑及文件名，json文件形式
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
