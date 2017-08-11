# -*- coding: utf-8 -*-
"""
@version: 4.1
@author: Conley.K
@license: Apache Licence 
@contact: conley.kong@gmail.com
@software: PyCharm
@file: execise.py
@time: 2017/7/13 0013 10:27
@desc: 为inference增加指定各层(共3层)的卷积核大小的接口
"""
import tensorflow as tf
import os
import hsi_inference
import DatasetKits as dk
import time

# 抑制提示信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SUMMARY_FLAG = False
MODULE_FLAG = False
TRAINING_RATIO = 0.01  # 百分数，比如0.8代表80%
NEIGHBOR = 1  # 近邻数，比如1代表单像元，4代表四近邻
BATCH_SIZE = 100  # 批大小默认是100
TRAINING_STEPS = 15000  # 默认训练步数：10000
REGULARAZTION_RATE = 0.0001  # 默认是0.0001
# ADAM_LEARNING_RATE_LIST = [5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 10e-4, 11e-4, 12e-4, 13e-4, 14e-4, 15e-4, 16e-4, 17e-4, 18e-4,19e-4, 20e-4, 21e-4, 22e-4, 23e-4, 24e-4, 25e-4]
ADAM_LEARNING_RATE_LIST = [10e-4, 11e-4, 12e-4, 13e-4, 14e-4, 15e-4, 16e-4, 17e-4, 18e-4,19e-4,20e-4, 21e-4, 22e-4, 23e-4, 24e-4, 25e-4,26e-4,27e-4,28e-4,29e-4,30e-4,]
# ADAM_LEARNING_RATE_LIST = [12e-4]
DATA_PATH = "../DATA/KSC"  # 读取实验数据的路径
BASE_LOG_PATH = "./KSC_Log_Repo_5.0/"
Version_Flag = "8,32,128,256_64,batch_100"
""" #放弃的参数
    # LEARNING_RATE_BASE = 0.15    #默认是0.1
    # LEARNING_RATE_DECAY = 0.99  #默认是0.99,指数下降时才会用
    # ADAM_LEARNING_RATE = 4e-4   #AdamOptimizer学习率默认为1e-4，要比GradientDescentOptimizer高效
    # LOG_PATH = "H:/CodeFarm/Python/TensorFlow/KSC_Log_Repo_Gt1.2/"+VERSION_FLAG    #保存日志文件的路径
    # VERSION_FLAG = "4_16x11"
"""


def main(_):
    # mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)
    # mnist = dk.load_ksc(DATA_PATH)
    mnist = dk.load_preped_ksc(data_dir=DATA_PATH, train_ratio=TRAINING_RATIO, neighbor=NEIGHBOR)  # 载入预处理的数据
    with tf.name_scope("preparations"):
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # dropout keepprob
        # regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)  # regularizer 加入正则化后开始的收敛会较慢,精确度会降低
        regularizer = None;
        x = tf.placeholder(tf.float32, [None, hsi_inference.IMAGE_HEIGHT_SIZE * hsi_inference.IMAGE_WIDTH_SIZE],
                           name="x")
        y_ = tf.placeholder(tf.float32, [None, hsi_inference.LABEL_SIZE], name="y_")  # y_应该是一个14列的数组
        #日志相关操作
        # log_path = BASE_LOG_PATH + unique_name
        log_path = BASE_LOG_PATH + str(TRAINING_RATIO * 100) + "_" + str(NEIGHBOR) + "_" + str(
            hsi_inference.IMAGE_HEIGHT_SIZE) + "x" + str(
            hsi_inference.IMAGE_WIDTH_SIZE) + "_" + Version_Flag
        # if (regularizer == None):
        #     log_path = os.path.join(log_path, "no_regularizer")
        # if (SUMMARY_FLAG == False):
        #     log_path = os.path.join(log_path, "no_summmary")
        if (os.path.exists(log_path) == False):
            os.makedirs(log_path)
        TEST_RESULT_FILE = BASE_LOG_PATH + "/" + str(TRAINING_RATIO * 100) + "_" + str(NEIGHBOR) + "_" + str(
            hsi_inference.IMAGE_HEIGHT_SIZE) + "x" + str(
            hsi_inference.IMAGE_WIDTH_SIZE) + "_" + Version_Flag + "test_accuracy_summary.csv"
        test_result_file = open(TEST_RESULT_FILE, 'a')
        test_result_file.write("学习率,\t精确度,\t训练用时,\t测试用时\n")  # 写入表头
        test_result_file.close()


    with tf.name_scope("inference_ops"):
        y = hsi_inference.inference(x, keep_prob, regularizer, neighbor=NEIGHBOR)

    # loss functions
    with tf.name_scope("loss_function"):
        # cross_entropy
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        if regularizer != None:
            regular_losses = tf.get_collection('regularizer_losses')
            regular_losses_sum = tf.add_n(regular_losses)  # add_n可以将多个同形同性的tensor合并为一个同形同性的tensor
            loss = cross_entropy + regular_losses_sum
        else:
            loss = cross_entropy

    # accuracy
    with tf.name_scope("accuracy_reckons"):
        with tf.name_scope("correct_prediction"):
            # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar(name="accuracy", tensor=accuracy)

    """
        和TensorFlow中其他操作类似，tf.summary.scalar、tf.summary.histogram和tf.summary.image函数
        不会立即执行，需要通过sess.run来明确调用这些函数.
        由于程序中定义的写日志操作比较多，一一调用比较麻烦所以TensorFlow提供了tf.merge_all_summaries函数来整理所有日志的生成操作。
        在TensorFlow程序执行的过程中只需要运行这个操作就可以将代码定义的所有日志生成操作执行一次，从而将所有日志写入文件
    """

    for adam_learning_rate in ADAM_LEARNING_RATE_LIST:
        print("学习率：",adam_learning_rate)
        with tf.name_scope("main_initialization"):
            unique_name = str(TRAINING_RATIO * 100) + "_" + str(NEIGHBOR) + "_" + str(adam_learning_rate) + "_" + str(hsi_inference.IMAGE_HEIGHT_SIZE) + "x" + str(hsi_inference.IMAGE_WIDTH_SIZE)
            train_result_path = log_path + "/" + unique_name + ".csv"
            train_result_file = open(train_result_path, 'a')
            train_log = []  # 用于缓存所有训练日志
            train_log.append("步数,精度\n")
            start_time = time.time()  # 记录开始时间
        # summary writers
        if (MODULE_FLAG):
            with tf.name_scope("summary_writers"):
                train_writer = tf.summary.FileWriter(log_path + '/train')
                test_writer = tf.summary.FileWriter(log_path + '/test')
                merge_ops = tf.summary.merge_all()  # 每次运行merge_ops的时候都会往事件文件中写入最新的即时数据，其输出会调用writer的add_summary函数

        # train step
        with tf.name_scope("train_step"):
            train_step = tf.train.AdamOptimizer(adam_learning_rate).minimize(loss)
            # train_step = tf.train.AdamOptimizer(ADAM_LEARNING_RATE).minimize(loss)
            # train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE_BASE).minimize(loss)

        # sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
        # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()  # 初始化TensorFlow持久化类
            with tf.name_scope("training_steps"):
                if (MODULE_FLAG):
                    train_writer.add_graph(sess.graph)
                for i in range(TRAINING_STEPS):
                    batch = mnist.train.next_batch(BATCH_SIZE)
                    if i % BATCH_SIZE == 0:  # 每个batch保存一次模型并记录运行状态
                        with tf.name_scope("checkpoint_train_step"):
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)  # 配置运行时需要记录的信息
                            run_metadata = tf.RunMetadata()  # 运行时记录运行信息的proto
                            # 将配置信息和记录运行信息的proto传入运行的过程，从而记录路运行时每一个节点的时间空间开销信息
                            if (SUMMARY_FLAG):
                                summary, train_accuracy = sess.run([merge_ops, accuracy],
                                                                   feed_dict={x: batch[0], y_: batch[1],
                                                                              keep_prob: 1.0},
                                                                   run_metadata=run_metadata, options=run_options)
                                train_writer.add_summary(summary, global_step=i)  # 将当前的TensorFlow计算图写入日志。
                            else:
                                train_accuracy = sess.run(accuracy,
                                                          feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0},
                                                          run_metadata=run_metadata, options=run_options)

                            if (MODULE_FLAG):
                                train_writer.add_run_metadata(run_metadata, tag='第%03d步' % i,
                                                              global_step=i)  # 将节点的运行时信息写入日志文件
                                # 保存当前的模型。需要注意的是此处给出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数，如simple_minist_model.ckpt-1000表示训练1000轮后的模型
                                # saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), i)
                                saver.save(sess, os.path.join(log_path, "model.ckpt"), i)
                            print("Step %d, training accuracy %g" % (i, train_accuracy))
                            # train_result_file.write("步数: %d, 精度: %g \n" % (i, train_accuracy))
                            train_log.append("%d, %g \n" % (i, train_accuracy))

                    # 正常训练步
                    with tf.name_scope("normal_train_step"):
                        if (SUMMARY_FLAG):
                            summary, _ = sess.run([merge_ops, train_step],
                                                  feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                            train_writer.add_summary(summary, global_step=i)  # 将当前的TensorFlow计算图写入日志。
                        else:
                            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

                end_time = time.time()
                training_time = end_time - start_time
                # 开始日志文件的磁盘写入
                for item in train_log:
                    train_result_file.write(item)
                train_result_file.close()

            with tf.name_scope("test_steps"):
                if (MODULE_FLAG):
                    test_writer.add_graph(sess.graph)
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                test_start_time = time.time()
                if (SUMMARY_FLAG):
                    summary, real_acccuracy = sess.run([merge_ops, accuracy],
                                                       feed_dict={x: mnist.test.images, y_: mnist.test.labels,
                                                                  keep_prob: 1.0}, options=run_options,
                                                       run_metadata=run_metadata)
                    test_writer.add_summary(summary)
                else:
                    real_acccuracy = sess.run(accuracy,
                                              feed_dict={x: mnist.test.images, y_: mnist.test.labels,
                                                         keep_prob: 1.0},
                                              options=run_options, run_metadata=run_metadata)
                test_end_time = time.time()
                if (MODULE_FLAG):
                    test_writer.add_run_metadata(run_metadata, tag='测试操作')  # 将节点的运行时信息写入日志文件
                test_time = test_end_time - test_start_time
                # 将测试结果写入文件中保存
                test_result_file = open(TEST_RESULT_FILE, 'a')
                try:
                    print("test accuracy %g" % real_acccuracy)
                    _results = str(adam_learning_rate) + ",\t" + str(real_acccuracy) + ",\t" + str(training_time)+ ",\t" + str(test_time) + " \n "
                    test_result_file.write(_results)
                finally:
                    test_result_file.close()

            if (MODULE_FLAG):
                train_writer.close()
                test_writer.close()

    print("---------------programs finished -------------")

if __name__ == '__main__':
    tf.app.run(main=main)
