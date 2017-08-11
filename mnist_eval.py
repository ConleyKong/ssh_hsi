# -*- coding: utf-8 -*-
"""
@version: ??
@author: Conley.K
@license: Apache Licence 
@contact: conley.kong@gmail.com
@software: PyCharm
@file: mnist_eval.py
@time: 2017/7/10 0010 16:51
@desc: 
"""
import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.examples.tutorials.mnist import input_data
#加载mnist_inference.py和mnist_train.py中定义的常量和函数
import mnist_inference
import mnist_train
#抑制提示信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#加载最新模型的时间间隔，默认为10s，加载后使用测试数据集测试得到最新模型的正确率
EVAL_INTERVAL_SECS = 10

#验证数据集的大小，默认为5000
VALIDATION_NUM = 5000

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        #定义输入输出的格式:
        # x = tf.placeholder(dtype=tf.float32,shape = [None,mnist_inference.INPUT_NODE],name="x-input")
        # 重定义图片的格式
        x = tf.placeholder(dtype=tf.float32, shape=[VALIDATION_NUM,mnist_inference.IMAGE_HEIGHT_SIZE,mnist_inference.IMAGE_WIDTH_SIZE,mnist_inference.NUM_CHANNELS], name="x-input")
        y_ = tf.placeholder(dtype=tf.float32,shape=[None,mnist_inference.NUM_LABELS],name='y-input')

        reshaped_x = np.reshape(mnist.validation.images,[VALIDATION_NUM,mnist_inference.IMAGE_HEIGHT_SIZE,mnist_inference.IMAGE_WIDTH_SIZE,mnist_inference.NUM_CHANNELS])
        # validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        validate_feed = {x: reshaped_x, y_: mnist.validation.labels}

        #直接通过调用封装好的函数来计算前向传播的结果,保存在y中
        # 由于测试时无需关心正则化损失的值，因此此处正则化损失函数为None
        y = mnist_inference.inference(x,regularizer=None,train = False)

        #使用前向传播的结果计算正确率。
        # 如果需要对未知的样例进行分类，那么使用tf.argmax(y,1)就可以得到输入样例的预测类别了
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))       #判断预测值是否和真实分类相同，相同则那个位置为1
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))   #将correct_prediction的value转换为float32类型然后将这个张量进行reduce_mean操作

        #通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来取平均值了。这样就可以完全公用mnist_inference.py中定义的前向传播过程
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        #每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的变化
        while True:
            with tf.Session() as sess:
                #tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    #加载模型
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    #通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
                    print("经过%s步的训练，验证精确度为%g" %(global_step,accuracy_score))
                else:
                    print("未找到检查点文件")
                    return

                #再经过一个时间段后重新计算
                time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets(mnist_train.DATA_PATH,one_hot=True,validation_size=VALIDATION_NUM)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()