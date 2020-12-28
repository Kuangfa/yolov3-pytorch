#-*-encoding:utf-8-*-
"""
# function/功能 : 
# @File : makeTxt.py 
# @Time : 2020/9/11 11:37 
# @Author : kf
# @Software: PyCharm
"""
import os
import random
data_dir='data/'

trainval_percent = 0.2
train_percent = 0.8
xmlfilepath = data_dir+'Annotations'
txtsavepath = data_dir+'ImageSets/Main'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
train_val = random.sample(list, tv)
train_val_test = random.sample(train_val, tr)

ftrainval = open(data_dir+'ImageSets/Main/trainval.txt', 'w')
ftest = open(data_dir+'ImageSets/Main/test.txt', 'w')
ftrain = open(data_dir+'ImageSets/Main/train.txt', 'w')
fval = open(data_dir+'ImageSets/Main/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in train_val:
        ftrainval.write(name)
        if i in train_val_test:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()