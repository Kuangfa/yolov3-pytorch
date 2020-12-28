# -*-encoding:utf-8-*-
"""
# function/功能 : 
# @File : voc_label.py 
# @Time : 2020/9/11 11:32 
# @Author : kf
# @Software: PyCharm
"""
import os  # os 模块 提供了非常丰富的方法用来处理文件和目录。
import xml.etree.ElementTree as ET  # 给包xml.etree.ElementTree 定义一个 ET 别名  操作XML文件的包
from os import getcwd  # 从os包中引入 listdir, getcwd 类

sets = ['train', 'test','val']
classes = ["RBC"]  # 训练的类别，只有一个类别
data_dir='data/'

# -----------------------函数定义开始------------------
def convert(size, box):  #
    '''

    :param size:    图片大小
    :param box:     图片标注的坐标，分别为 xmin,xmax,ymin,ymax
    :return:        先将图像变成1*1大小，之后返回标注中心坐标与对应宽度与高度，分别为x,y,w,h
    '''
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation( image_id):
    in_file = open(data_dir+'Annotations/%s.xml' % (image_id), encoding='UTF-8')
    out_file = open(data_dir+'labels/%s.txt' % (image_id), 'w', encoding='UTF-8')
    """
        ‘w’打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。
    """
    # 从xml文件中获取图片标注的宽与高
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


# -----------------------函数定义结束------------------

wd = getcwd()  # os.getcwd() 方法用于返回当前工作目录。
print(wd)
for image_set in sets:
    if not os.path.exists(data_dir+'labels/'):
        os.mkdir(data_dir+'labels/')
    image_ids = open(data_dir+'ImageSets/Main/%s.txt'%(image_set), encoding='UTF-8').read().strip().split()  # 获取数字，以便取图片
    """
            # read() 方法用于从文件读取指定的字节数，如果未给定或为负则读取所有。
            # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
            # split() 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则仅分隔 num 个子字符串
    """
    list_file = open(data_dir+'%s.txt' % (image_set), 'w', encoding='UTF-8')
    for image_id in image_ids:
        # list_file.write('%s/JPEGImages/%s.jpg\n' % (wd, image_id))
        list_file.write(data_dir+'JPEGImages/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()
