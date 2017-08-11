# -*- coding: utf-8 -*-
"""
@version: ??
@author: Conley.K
@license: Apache Licence 
@contact: conley.kong@gmail.com
@software: PyCharm
@file: mnist_train.py
@time: 2017/7/10 0010 10:45
@desc: 根据mnist_version1.0改造而来，依据YannLecun论文中提出的网络模型实现
"""
import tensorflow as tf
import numpy as np
import os
#抑制提示信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#输入数据：
from tensorflow.examples.tutorials.mnist import input_data

#载入mnist_inference.py中定义的常量和前向传播函数
import mnist_inference

#配置神经网络参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8  #默认是0.8
LEARNING_RATE_DECAY = 0.8  #默认是0.99
REGULARAZTION_RATE = 0.00001 #默认是0.0001
TRAINING_STEPS = 20000     #训练轮数，默认是30000轮
MOVING_AVERAGE_DECAY = 0.9

#模型保存的路径和文件名
MODEL_SAVE_PATH = "./model_repo/"
MODEL_NAME = "simple_minist_model.ckpt"
MODEL_LENGTH = 1000     #每1000轮保存一次模型数据

#存放训练数据的路径
DATA_PATH = "I:/MNIST/DATA";

#保存日志文件的路径
LOG_PATH = "H:/CodeFarm/Python/TensorFlow/Log_repo/LeNet1.0"

#训练函数
def train(mnist):
    #TODO 由于卷积神经网络的输入层为一个三维矩阵，因此将x的占位符进行了调整
    #x = tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name='x-input')
    #此处的输入数据的shape需要为四维矩阵，其中第一维表示batch的大小，第二、三维为图片的高宽，第四维为图片的深度，比如RGB
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_inference.IMAGE_HEIGHT_SIZE,mnist_inference.IMAGE_WIDTH_SIZE,mnist_inference.NUM_CHANNELS], name='x-input')
        # 定义真实值的placeholder
        y_ = tf.placeholder(tf.float32,[None,mnist_inference.NUM_LABELS],name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # regularizer = None;       #不使用正则化时结果左右摇摆，损失函数的下降不明显
    #调用mnist_inference.py中定义的前向传播过程,得到预测的结果
    y = mnist_inference.inference(x,train=True,regularizer=regularizer)
    global_step = tf.Variable(0,trainable=False)

    # 滑动平均计算
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 损失函数的计算
    with tf.name_scope("loss_function"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,1),logits=y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        if regularizer != None:
            regular_losses = tf.get_collection('losses')
            regular_losses_sum = tf.add_n(regular_losses)  #add_n可以将多个同形同性的tensor合并为一个同形同性的tensor
            loss = cross_entropy_mean+regular_losses_sum
        else:
            loss = cross_entropy_mean

    #定义学习率、优化算法以及具体的训练工作
    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step)

        with tf.control_dependencies([train_step,variable_averages_op]):
            train_op = tf.no_op(name="train")

        #初始化TensorFlow持久化类
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            tf.global_variables_initializer().run()

            # 生成写日志的writer并将当前的TensorFlow计算图写入日志。TensorFlow提供了多种写日志文件的API
            train_writer = tf.summary.FileWriter(LOG_PATH+'/train',sess.graph)

            #训练过程不再测试模型在验证数据上的表现，验证和测试过程将会有一个独立的程序完成
            for i in range(TRAINING_STEPS+1):
                with tf.name_scope("once_train"):
                    xs,ys = mnist.train.next_batch(BATCH_SIZE)
                    #TODO 需要将输入的训练数据格式调整为一个四维矩阵
                    reshaped_xs = np.reshape(xs,[BATCH_SIZE,mnist_inference.IMAGE_HEIGHT_SIZE,mnist_inference.IMAGE_WIDTH_SIZE,mnist_inference.NUM_CHANNELS])

                    #TODO 每MODEL_LENGTH轮保存一次模型并记录运行状态
                    if i % MODEL_LENGTH == 0:
                        #配置运行时需要记录的信息
                        run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
                        #运行时记录运行信息的proto
                        run_metadata = tf.RunMetadata()
                        #将配置信息和记录运行信息的proto传入运行的过程，从而济洛路运行时每一个节点的时间空间开销信息
                        _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:reshaped_xs,y_:ys},options=run_options,run_metadata=run_metadata)
                        #将节点的运行时信息写入日志文件
                        train_writer.add_run_metadata(run_metadata,'第%03d步'%i)
                        #在控制台输出当前的训练情况：包含当前batch上的损失大小。通过损失函数大小可以大概了解训练的情况。对于验证数据集上的正确率信息会有一个单独的程序来生成
                        print("经过%d个训练步骤，批量训练中的损失值为%g "%(step,loss_value))
                        #保存当前的模型。需要注意的是此处给出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数，如simple_minist_model.ckpt-1000表示训练1000轮后的模型
                        saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step)
                    else:
                        _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            train_writer.close()

def main(argv=None):
    mnist = input_data.read_data_sets(DATA_PATH,one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()