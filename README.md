## YOLOV4：You Only Look Once目標檢測模型在Keras當中的實現
---

### 目錄
1. [性能情況 Performance](#性能情況)
2. [實現的內容 Achievement](#實現的內容)
3. [所需環境 Environment](#所需環境)
4. [注意事項 Attention](#注意事項)
5. [小技巧的設置 TricksSet](#小技巧的設置)
6. [文件下載 Download](#文件下載)
7. [預測步驟 How2predict](#預測步驟)
8. [訓練步驟 How2train](#訓練步驟)

### 性能情況
| 訓練資料集 | 權值檔案名稱 | 測試資料集 | 輸入圖片大小 | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| VOC07+12+COCO | [yolo4_voc_weights.h5](https://github.com/bubbliiiing/yolov4-keras/releases/download/v1.0/yolo4_voc_weights.h5) | VOC-Test07 | 416x416 | - | 86.4
| COCO-Train2017 | [yolo4_weight.h5](https://github.com/bubbliiiing/yolov4-keras/releases/download/v1.0/yolo4_weight.h5) | COCO-Val2017 | 416x416 | 43.1 | 66.0

### 實現的內容
- [x] 主幹特徵提取網路：DarkNet53 => CSPDarkNet53
- [x] 特徵金字塔：SPP，PAN
- [x] 訓練用到的小技巧：Mosaic資料增強、Label Smoothing平滑、CIOU、學習率余弦退火衰減
- [x] 啟動函數：使用Mish啟動函數


### 所需環境
tensorflow-gpu==1.13.1  
keras==2.1.5  

### 注意事項
代碼中的yolo4_weights.h5是基於608x608的圖片訓練的，但是由於顯存原因。我將代碼中的圖片大小修改成了416x416。有需要的可以修改回來。 代碼中的默認anchors是基於608x608的圖片的。   
**注意不要使用中文標籤，資料夾中不要有空格！**   
**在訓練前需要務必在model_data下新建一個txt文檔，文檔中輸入需要分的類，在train.py中將classes_path指向該文件**。  

### 小技巧的設置
在train.py文件下：   
1、mosaic參數可用於控制是否實現Mosaic資料增強。   
2、Cosine_scheduler可用於控制是否使用學習率余弦退火衰減。   
3、label_smoothing可用於控制是否Label Smoothing平滑。

### 文件下載
訓練好的權重可至google雲端下載trained_weights.h5
連結: https://drive.google.com/drive/folders/1xL1o9awOJmTygzEHJ8cRzH__QpDz7dt0?usp=sharing 


### 預測步驟
#### 1、使用預訓練權重
a、下載完庫後解壓，在google雲端(https://drive.google.com/drive/folders/1xL1o9awOJmTygzEHJ8cRzH__QpDz7dt0?usp=sharing)
下載trained_weights.h5後，放入model_data，運行predict.py，輸入  
```python
img/street.jpg
```
可完成預測。  
b、利用video.py可進行攝像頭檢測。  
#### 2、使用自己訓練的權重
a、按照訓練步驟訓練。  
b、在yolo.py檔裡面，在如下部分修改model_path和classes_path使其對應訓練好的檔；**model_path對應logs資料夾下面的權值文件，classes_path是model_path對應分的類**。  
```python
_defaults = {
    "model_path": 'model_data/yolo4_weight.h5',
    "anchors_path": 'model_data/yolo_anchors.txt',
    "classes_path": 'model_data/coco_classes.txt,
    "score" : 0.5,
    "iou" : 0.3,
    # 顯存比較小可以使用416x416
    # 顯存比較大可以使用608x608
    "model_image_size" : (416, 416)
}

```
c、運行predict.py，輸入  
```python
img/street.jpg
```
可完成預測。  
d、利用video.py可進行攝像頭檢測。  

### 訓練步驟
1、本文使用VOC格式進行訓練。  
2、訓練前將標籤文件放在VOCdevkit資料夾下的VOC2007資料夾下的Annotation中。  
3、訓練前將圖片檔放在VOCdevkit資料夾下的VOC2007資料夾下的JPEGImages中。  
4、在訓練前利用voc2yolo4.py檔生成對應的txt。  
5、再運行根目錄下的voc_annotation.py，運行前需要將classes改成你自己的classes。**注意不要使用中文標籤，資料夾中不要有空格！**   
```python
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
```
6、此時會生成對應的2007_train.txt，每一行對應其**圖片位置**及其**真實框的位置**。  
7、**在訓練前需要務必在model_data下新建一個txt文檔，文檔中輸入需要分的類，在train.py中將classes_path指向該文件**，示例如下：   
```python
classes_path = 'model_data/new_classes.txt'    
```
model_data/new_classes.txt檔內容為：   
```python
cat
dog
...
```
8、運行train.py即可開始訓練。

### mAP目標檢測精度計算更新
更新了get_gt_txt.py、get_dr_txt.py和get_map.py文件。  
get_map文件克隆自https://github.com/Cartucho/mAP  
具體mAP計算過程可參考：https://www.bilibili.com/video/BV1zE411u7Vw

 

