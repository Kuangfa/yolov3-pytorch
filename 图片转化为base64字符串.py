#-*-encoding:utf-8-*-
"""
# function/功能 : 
# @File : 图片转化为base64字符串.py 
# @Time : 2020/12/28 10:18 
# @Author : kf
# @Software: PyCharm
"""
import base64
f=open('samples/000000.png','rb') #二进制方式打开图文件
ls_f=base64.b64encode(f.read()) #读取文件内容，转换为base64编码
f.close()
print(ls_f)