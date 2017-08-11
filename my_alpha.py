# -*- coding: utf-8 -*-
"""
@version: ??
@author: Conley.K
@license: Apache Licence 
@contact: conley.kong@gmail.com
@software: PyCharm
@file: my_alpha.py
@time: 2017/7/10 0010 23:48
@desc: 
"""
import tensorflow as tf
import numpy as np
import os

#抑制提示信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# t = tf.constant([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
# # print(t.get_shape())
# # print(t.get_shape().as_list())
# with tf.Session() as sess:
#     print(sess.run(tf.reshape(t, [2,-1])))

#测试双循环下内部循环的变量是否会自动归位 ☑ 会自动归位
# l1 = range(4)
# l2 = range(9)
# for i in l1:
#     for j in l2:
#         print(i,"x",j,"=",i*j)

#测试从字典获取参数并合并
IO_conv1 = [1,32]
IO_conv2 = [32,128]
IO_conv3 = [128,512]
size_mapper = {'3x3':[3, 3], '2x2':[2, 2], '1x1':[1, 1], '3x2':[3, 2], '2x3':[2, 3]}
filter_units = [['3x3', '3x3', '3x3']
                , ['3x2','3x2','3x2']
                , ['2x3','2x3','2x3']
                , ['2x2','2x2','2x2']
                , ['3x2','2x3','3x2']
                , ['2x3','3x3','2x3']
                , ['3x3','2x2','1x1']
                , ['1x1','2x2','3x3']
                , ['3x3','1x1','3x3']
                , ['3x3','2x2','3x3']]
for unit in filter_units:
    conv1_shape = size_mapper.get(unit[0]) + IO_conv1
    conv2_shape = size_mapper.get(unit[1]) + IO_conv2
    conv3_shape = size_mapper.get(unit[2]) + IO_conv3
    print(str(unit[0])+"_"+str(unit[1])+"_"+str(unit[2]))
    print(conv1_shape)
    print(conv2_shape)
    print(conv3_shape)


#测试list合并
# a = [3,3]
# b = [4,6]
# c = a+b
# print(c)

#测试字典参数
# def dict_function(para_dict=None):
#     filter_dict={"conv1_shape":(3,3),"conv2_shape":(3,3),"conv3_shape":(3,3)}
#     if(para_dict!=None):
#         filter_dict.update(para_dict)
#     for key in filter_dict.keys():
#         print(filter_dict[key])
#
# if __name__ == '__main__':
#     dict_function({"conv1_shape":(3,2),"conv2_shape":(2,3),"conv3_shape":(3,2)})

#取模运算
# print(178%20)

#矩阵拓展
# a = np.array([[1,2,3,4,5,6,7,8,9],[9,8,7,6,5,4,3,2,1]])
# appendix = np.zeros([2,1])
# print("a:",a)
# print("appendix:",appendix)
# b = np.append(a,appendix,axis=1)
# print("扩展后：",b)