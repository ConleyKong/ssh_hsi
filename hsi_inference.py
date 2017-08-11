# -*- coding: utf-8 -*-
"""
@version: ??
@author: Conley.K
@license: Apache Licence 
@contact: conley.kong@gmail.com
@software: PyCharm
@file: cnn_inference.py
@time: 2017/7/13 0013 10:45
@desc:
    改变网络卷积层结构：
    1.卷积层的过滤深度分别调整为16和128
    2.卷积层的卷积核大小分别调整为5x5与3x3
    3.此时卷积结果为3x2x128=640,因此全连接层选取120
    4.增加第三层卷积层，卷积核大小为3x2，深度为512,结果是512的向量
    5.固定二维化的图片宽度为4xConv1_size,列自动推理
"""
import tensorflow as tf
import numpy as np
import os
#抑制提示信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#波段信息一共176 = 16x11

INPUT_CHANNEL = 1
LABEL_SIZE = 14

CONV1_SIZE = 3          #默认第一层卷积层的大小，其乘以4就是默认的二维化频谱信息的宽度
CONV1_DEPTH = 8
CONV2_SIZE = 3
CONV2_DEPTH = 32
CONV3_SIZE = 3
CONV3_DEPTH = 128

FULL1_SIZE = 256
FULL2_SIZE = 64
IMAGE_WIDTH_SIZE = 20
IMAGE_HEIGHT_SIZE = 9

#TODO 为不同的近邻策略设计合适的传播方式
"""
    @:param input_tensor    卷积操作的tensor对象
    @:param keep_prob       dropout的保留概率
    @:param regularizer     正则化方式
    @:param shape_dict      各卷积层的卷积核定义，字典形式，如{"conv1_shape":[height,width,input_channel,output_channel],"conv2_shape":[height2,width2,i_channel2,o_channel2,"conv3_shape":……]}
"""
def inference(input_tensor, keep_prob=1,regularizer=None,neighbor=1):

    with tf.name_scope("input_reshape"):
        """
            将输入的向量还原成图片的像素矩阵并通过tf.summary.image函数定义将当前图片信息写入日志操作
            shape为批大小，高，宽，频道数量
        """
        input_image = tf.reshape(input_tensor, [-1, IMAGE_HEIGHT_SIZE, IMAGE_WIDTH_SIZE, INPUT_CHANNEL])
        tf.summary.image(name="input",tensor=input_image,max_outputs=10)
        conv1_shape = [CONV1_SIZE, CONV1_SIZE, INPUT_CHANNEL, CONV1_DEPTH]
        conv2_shape = [CONV2_SIZE, CONV2_SIZE, CONV1_DEPTH, CONV2_DEPTH]
        conv3_shape = [CONV3_SIZE,CONV3_SIZE,CONV2_DEPTH,CONV3_DEPTH]

    pool1 = get_cnn_layer(input_image, conv1_shape, layer_name="conv1_layer")
    pool2 = get_cnn_layer(pool1, conv2_shape, layer_name="conv2_layer")
    pool3 = get_cnn_layer(pool2,conv3_shape,layer_name="conv3_layer")

    #经过三次卷积三次最大池化层图像被压缩成了batch*1*1*512的shape,使用get_shape获得各个维度的数据其中shape[0]为batch大小
    with tf.name_scope("flat_layer"):
        final_pool = pool3
        final_pool_shapes = final_pool.get_shape().as_list()
        final_nodes = final_pool_shapes[1]*final_pool_shapes[2]*final_pool_shapes[3]
        flated_final_pool = tf.reshape(final_pool, [-1, final_nodes])

    #第一个全连接层，使用激活函数tf.nn.relu
    full1 = get_fc_layer(input_tensor=flated_final_pool,output_num=FULL1_SIZE,layer_name="full1_layer",act=tf.nn.relu,regularizer=regularizer)
    #第二个全连接层，使用激活函数tf.nn.relu
    final_full = get_fc_layer(input_tensor=full1,output_num=FULL2_SIZE,layer_name="full2_layer",act=tf.nn.relu,regularizer=regularizer)
    # dropout
    with tf.name_scope("dropout_layer"):
        fulls_dropout = tf.nn.dropout(final_full, keep_prob)
    #最后一个全连接层,不使用激活函数
    y = get_fc_layer(input_tensor=fulls_dropout,output_num=LABEL_SIZE,layer_name="output_layer",regularizer=regularizer)
    return y

#生成一层卷积神经网络
#   @param input_tensor 也为一个四维tensor：[-1, IMAGE_HEIGHT_SIZE, IMAGE_WIDTH_SIZE, INPUT_CHANNEL]
#   @param filter_weight 为一个四维tensor，[height,width,input_depth,output_depth]
#   @return 返回卷积的结果
def get_cnn_layer(input_tensor, filter_dim, layer_name, act=tf.nn.relu):
    print(layer_name,"卷积核尺寸: ",filter_dim)
    output_depth = filter_dim[3]
    # print(output_depth)
    with tf.name_scope(layer_name):
    #将同一层神经网络放在一个统一的命名空间下
        with tf.name_scope("weights"):
            conv_weights = get_kernel_weights(filter_dim)
            # variable_statistics_summaries(name=layer_name+'/weights', values=conv_weights)
        with tf.name_scope("bias"):
            conv_bias = get_bias_varibales([output_depth])
            # variable_statistics_summaries(name=layer_name+'/bias', values=conv_bias)
        with tf.name_scope("pre_activate"):
            preactivate = conv2d(input_tensor, conv_weights) + conv_bias
            # tf.summary.histogram(name=layer_name+'/pre_activate',values=preactivate)
        with tf.name_scope("post_activate"):
            postactivate = act(preactivate,name='activation')
            # tf.summary.histogram(name=layer_name+'/post_activate',values=postactivate)
        with tf.name_scope("post_pool"):
            result = get_max_pool_2x2(postactivate)
            # tf.summary.histogram(name=layer_name + '/post_pool', values=result)
        return result

#生成一层全连接层网络
#   @param  input_tensor 输入的张量
#   @param  output_num  输出张量的维度
#   @param  layer_name  全连接层名称，也是namescope
#   @param  act         激活函数名称
def get_fc_layer(input_tensor,output_num,layer_name,act=None,regularizer=None):
    with tf.name_scope(layer_name):
    #首先将同一层全连接网络放在一个统一的命名空间中
        shape_list = input_tensor.get_shape().as_list()
        input_num = shape_list[-1]
        with tf.name_scope("weights"):
        #声明权重信息并调用生成权重监控信息日志函数
            weights = get_kernel_weights(shape=[input_num,output_num])
            # variable_statistics_summaries(name=layer_name+"/weights",values=weights)
            # 添加正则化损失
            if regularizer != None:
                tf.add_to_collection("regularizer_losses", regularizer(weights))
        with tf.name_scope("bias"):
        #声明偏置信息并调用生成权重监控信息日志函数
            bias = get_bias_varibales(shape=[output_num])
            # variable_statistics_summaries(name=layer_name+"/bias",values=bias)
        with tf.name_scope("pre_activate"):
        #计算激活前输出节点的数据
            preactivate = tf.matmul(input_tensor,weights)+bias
            # tf.summary.histogram(name=layer_name+'/pre_activations',values=preactivate)
            if (act == None):
                result = preactivate
            else:
                with tf.name_scope("post_activate"):
                    result = act(preactivate, name="activations")
                    # tf.summary.histogram(name=layer_name+'/activations',values=result)
        return result


#默认的pool使用2x2进行过滤
def get_max_pool_2x2(input_tensor, padding_type="SAME"):
    return tf.nn.max_pool(input_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding_type)

def get_kernel_weights(shape,mean=0.0,stddev=0.1):
    initial = tf.truncated_normal(shape,mean=mean,stddev=stddev)
    return tf.Variable(initial)

def get_bias_varibales(shape,init_value=0.1):
    initial = tf.constant(value=init_value, shape=shape)
    return tf.Variable(initial)

def conv2d(input_tensor, kernel_shape,padding_type="SAME"):
    #padding_type = 'SAME'是使用全零填充
    return tf.nn.conv2d(input_tensor, kernel_shape, strides=[1, 1, 1, 1], padding=padding_type)

# @desc 生成变量监控信息并定义生成监控信息日志的操作
# @para
#   var:需要记录的值
#   name:可视化中显示的名字
def variable_statistics_summaries(name, values):
    #将生成监控信息的操作放到同一命名空间中：
    with tf.name_scope("summaries"):
        #记录元素取值分布
        '''
            通过tf.summary.histogram函数记录张量中元素的取值分布。对于给出的图表名称和张量，tf.summary.histogram函数会升恒一个Summary protocol buffer。
            将summary写入TensorBoard日志文件后可以再HISTOGRAM栏下看到对应名称的图表。
            与其他操作类似，tf.summary.histogram函数不会立即被执行，只有当sess.run函数明确调用这个操作时TensorBoard才会真正生成并输出Summary protocol buffer
        '''
        tf.summary.histogram(name, values)

        #记录平均值
        """
            计算变量的平均值，定义生成平均值信息日志的操作。
            记录变量平均值信息的日志标签名为‘mean/’+name,其中name为命名空间，/是命名空间的分隔符
            相同命名空间中的监控指标会被整合到同一栏中，name则给出了当前监控指标属于那一个向量
        """
        mean = tf.reduce_mean(values)
        tf.summary.scalar('mean/'+name,mean)
        """
            #记录最小值
            min = tf.reduce_min(values)
            tf.summary.scalar('min/'+name,min)
    
            #记录最大值
            max = tf.reduce_max(values)
            tf.summary.scalar('max/'+name,max)
        """

        # 记录标准差
        """
            计算变量的标准差并定义生成其日志的操作
        """
        # stddev = tf.sqrt(tf.reduce_mean(tf.square(values - mean)))
        # tf.summary.scalar('stddev/'+name,stddev)




""" ## deprecated functions and variables
    ## first conv layer
    # with tf.name_scope("conv1_layer"):
    #     conv1_weights = get_kernel_weights([CONV1_SIZE, CONV1_SIZE, INPUT_CHANNEL, CONV1_DEPTH])
    #     conv1_bias = get_bias_varibales([CONV1_DEPTH])
    #     input_image = tf.reshape(input_tensor, [-1, IMAGE_HEIGHT_SIZE, IMAGE_WIDTH_SIZE, INPUT_CHANNEL])
    #     conv1 = tf.nn.relu(conv2d(input_image, conv1_weights) + conv1_bias)
    #     pool1 = max_pool_2x2(conv1)
    ## second conv layer
    # with tf.name_scope("conv2_layer"):
    #     conv2_weights = get_kernel_weights([CONV2_SIZE, CONV2_SIZE, CONV1_DEPTH, CONV2_DEPTH])
    #     conv2_bias = get_bias_varibales([CONV2_DEPTH])
    #     conv2 = tf.nn.relu(conv2d(pool1, conv2_weights) + conv2_bias)
    #     pool2 = max_pool_2x2(conv2)

    # first full connect layer
    # with tf.name_scope("full1_layer"):
    #     full1_weights = get_kernel_weights([final_nodes, FULL1_SIZE])
    #     full1_bias = get_kernel_weights([FULL1_SIZE])
    #     full1 = tf.nn.relu(tf.matmul(flated_final_pool, full1_weights) + full1_bias)
    #     #添加正则化loss
    #     if regularizer!=None:
    #         tf.add_to_collection("regularizer_losses",regularizer(full1_weights))
        # final output layer
    # with tf.name_scope("full2_layer"):
    #     full2_weights = get_kernel_weights([FULL1_SIZE, LABEL_SIZE])
    #     full2_bias = get_kernel_weights([LABEL_SIZE])
    #     y = tf.matmul(full1_dropout, full2_weights) + full2_bias
    #     # 添加正则化loss
    #     if regularizer != None:
    #         tf.add_to_collection("regularizer_losses", regularizer(full2_weights))
"""

