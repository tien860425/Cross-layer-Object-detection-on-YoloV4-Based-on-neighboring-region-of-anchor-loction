# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    :  
# @Author  : 
# @Email   :
# @File    : 
# --------------------------------------

coco_path="model_data/coco_classes.txt"
voc_path="model_data/voc_classes.txt"
with open(coco_path) as f:
    coco_lines = f.readlines()
with open(voc_path) as f:
    voc_lines = f.readlines()
tuplelist=[]
validid=[]
for i, line1 in enumerate(voc_lines):
    for j, line2 in enumerate(coco_lines):
        if line1.strip() == line2.strip():
            tp=(i,j)
            tuplelist.append(tp)
            validid.append(j)
            break

# anno_path= '../dataset/mscocoval.txt'
anno_path = '../dataset/mscocotrain.txt'

with open(anno_path) as f:
    lines = f.readlines()

validlist=[]
for cnt, line in enumerate(lines):
    temp = line.split()
    end = len(temp)
    hasvalid=False
    newline=[]
    newline.append(temp[0])
    for ii in range(1, end):
        box = temp[ii].split(',')
        for tt in tuplelist:
            if tt[1]==int(box[4]):
                box[4] = str(tt[0])
                newline.append(",".join(box))
                hasvalid = True
                break

    if hasvalid:
        aline=" ".join(newline)
        validlist.append(aline+'\n')

f = open('../dataset/mscoco_train20.txt', 'w')
# f = open('../dataset/mscoco_val20.txt', 'w')
f.writelines(validlist)
f.close()