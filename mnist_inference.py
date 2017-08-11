# -*- coding: utf-8 -*-
"""
@version: ??
@author: Conley.K
@license: Apache Licence 
@contact: conley.kong@gmail.com
@software: PyCharm
@file: mnist_inference.py
@time: 2017/7/10 0010 10:15
@desc: 构建LeNet的核心配置文件，实现了前向传播的过程
"""
import tensorflow as tf
import numpy as np
import os

#抑制提示信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#定义神经网络结构相关的参数
INPUT_NODE = 784

IMAGE_HEIGHT_SIZE = 28
IMAGE_WIDTH_SIZE=28
NUM_CHANNELS=1
NUM_LABELS = 10

#第一层卷积层的尺寸和深度
CONV1_DEEP=32
CONV1_SIZE=5
#第一层池化层的尺寸
POOL1_BATCH=1
POOL1_SIZE=2
POOL1_CHANNEL=1
#第二层卷积层尺寸和深度
CONV2_DEEP=64
CONV2_SIZE=5
#第二层池化层的尺寸
POOL2_BATCH=1
POOL2_SIZE=2
POOL2_CHANNEL=1
#全连接层节点个数
FULL1_SIZE = 512
KEEP_PROB = 0.9  #dropout方法中每一个元素的保留概率

PADDING = "VALID"

#重构卷积神经网络前向传播的过程。
#程序中增加dropout方法，可进一步提升模型可靠性并防止过拟合，dropout方法只用在训练过程中，
# 为了区分训练过程与测试过程我们增加了新的参数train用于区分训练过程和测试过程
# 输入参数input_tensor必须是4D数据：[batch, in_height, in_width, in_channels]
def inference(input_tensor,train=True, regularizer=None):

    ####################此处的卷积层不再加入正则化########################

    #声明第一层与输出层相连的卷积层的神经网络的变量并完成前向传播过程
    #通过不同命名空间隔离不同层变量，可以让每一层中的变量命名只需要考虑在当前层的作用无需担心重名问题。
    # 与标准的LeNet-5不同，这里定义的卷积层输入为28x28x1的原始MNIST图片。
    # 由于卷积层中使用了全0填充，因此输出为28x28x32
    ###-----------------声明第一层卷积层，使用variable_scope构造变量空间--------------------------###
    with tf.variable_scope("layer1-conv1"):
        # 此处使用tf.get_variable或tf.Variable没有本质区别，
        # 因为在训练或测试时没有在同一个程序中多次调用这个函数。
        # 如果在一个程序中多次调用则在第一次调用之后需要将reuse参数设置为True
        # conv1_weights = get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        conv1_weights = tf.get_variable("cnv1_weights",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        # biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        # 偏置系数的产生方式没有发生变化
        conv1_biases = tf.get_variable("cnv1_biases",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        #TODO 抛弃全连接形式的变换方式为卷积方式
        # layer1 = tf.nn.relu(tf.matmul(input_tensor,conv1_weights)+biases)
        # 增加了filter参数，strides参数(移动步长)，padding参数为SAME表明使用全零填充边缘,VALID表明不填充
        conv1 = tf.nn.conv2d(input_tensor,filter=conv1_weights,strides=[1,1,1,1],padding="SAME")
        # 套用激活函数，产生卷积最终结果，tf提供了nn下的conv与biases相加的处理函数
        conv1_relu = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))


    #声明第二层池化层前向传播过程。这里选用最大池化层。池化层输入为上一层的输出28x28x32,输出为14x14x32
    # 池化层过滤器边长为2，移动步长也为2，全零填充
    ###------------声明第二层池化层，使用name_scope构造变量空间----------------###
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(conv1_relu,ksize=[POOL1_BATCH,POOL1_SIZE,POOL1_SIZE,POOL1_CHANNEL],strides=[POOL1_BATCH,POOL1_SIZE,POOL1_SIZE,POOL1_CHANNEL],padding=PADDING)


    #声明第三层卷积层的变量并实现前向传播过程
    # 输入为14x14x32,输出为14x14x64或者10x10x32(valid)
    ###------------声明第三层卷积层，使用variable_scope构造变量空间--------------###
    with tf.variable_scope("layer3-conv2"):
        #构造滤波器
        conv2_weights = tf.get_variable("cnv2_weights",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.0))
        #构造偏置项
        conv2_biases = tf.get_variable("cnv2_biases",[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        #2D卷积操作
        conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding=PADDING)
        #添加激活函数
        conv2_relu = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))


    # 实现第四层池化层的前向传播过程，
    # 与第二层池化层的结构类似，输入为14x14x64或10x10x32(valid),输出为7x7x64或5x5x64(valid)
    ###------------声明第四层池化层结构，使用name_scope构造变量空间--------------###
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(conv2_relu,ksize=[POOL2_BATCH,POOL2_SIZE,POOL2_SIZE,POOL2_CHANNEL],strides=[POOL2_BATCH,POOL2_SIZE,POOL2_SIZE,POOL2_CHANNEL],padding=PADDING)


    #转换格式：将第四层池化层的输出转化为地我曾全连接层的输入格式。
    # 第四层的输出格式为7x7x64,第五层需要的输入为向量格式，因此我们这里需要将这个7x7x64的矩阵拉直成一个向量。
    # pool2.get_shape函数可以得到第四层输出矩阵的维度而不需要手工计算。
    # 需要注意的是，由于每一层神经网络的输入输出都为一个batch的矩阵，因此这里得到的维度也包含了一个batch中数据的个数
    ###--------------矩阵拉伸成向量------------------------###
    pool_shape = pool2.get_shape().as_list()
    # 计算矩阵拉伸成向量后的长度，该长度就是矩阵长度、宽度及深度的乘积，需要注意的是pool_shape[0]是一个batch中数据的个数
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    #通过tf.reshape函数将第四层的输出转化为一个batch的向量
    reshaped_nodes = tf.reshape(pool2,[pool_shape[0],nodes])


    #声明第五层全连接层的变量并实现前向传播过程。
    # 这一层的输入是拉直后的一组向量，长度为7x7x64=3136或5x5x64=1600,输出为一组长度512的向量。
    # 该层引入了dropout处理方式用来防止过拟合，从而可以使模型在测试数据上的效果更好，
    # dropout方式会在训练时随机将部分节点输出改为0。
    # dropout一般只在全连接层而不是卷积层或者池化层使用
    ###--------------声明第五层全连接层，使用variable_scope构建变量空间----------------------------###
    with tf.variable_scope("layer5-full1"):
        full1_weights = tf.get_variable("ful1_weights",[nodes,FULL1_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        #TODO 只有全连接层的权重需要正则化
        if regularizer != None:
            tf.add_to_collection("losses",regularizer(full1_weights))
        full1_biases = tf.get_variable("ful1_biases",[FULL1_SIZE],initializer=tf.constant_initializer(0.1))

        #计算全连接层的输出结果,此处不是使用的bias_add方法而是使用的最原始的矩阵乘
        # full1 = tf.nn.relu(tf.nn.bias_add(full1_weights,full1_biases))
        full1 = tf.nn.relu(tf.matmul(reshaped_nodes,full1_weights)+full1_biases)

        #TODO 区别对待训练和测试阶段
        if train:
            full1 = tf.nn.dropout(full1,keep_prob=KEEP_PROB)

    # 先前程序中的声明第二层神经网络变量(其实是输出层)并完成前向传播的代码，对比我们本程序中的卷积层可以发现这个其实是全连接层的样板
    # with tf.variable_scope("layer2"):
    #     conv1_weights = get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
    #     biases = tf.get_variable("biases",[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
    #     outlayer = tf.matmul(layer1,conv1_weights)+biases


    #声明第六层全连接层的变量并实现前向传播过程。
    # 该层的输入为一组长度为512的向量，输出为一组长度为10的向量，
    # 该层的输出通过softmax处理后得到最终的分类结果
    ###-------------声明第六层全连接层，使用variable_scope方式构建变量空间-------------###
    with tf.variable_scope("layer6-full2"):
        full2_weights = tf.get_variable("ful2_weights",[FULL1_SIZE,NUM_LABELS],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection("losses",regularizer(full2_weights))
        full2_biases = tf.get_variable("ful2_biases",[NUM_LABELS],initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(full1,full2_weights)+full2_biases

    #返回前向传播的最终逻辑结果
    return logit