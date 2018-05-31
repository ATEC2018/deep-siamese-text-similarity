#! /usr/bin/env python
#coding=utf-8
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
from input_helpers import InputHelper
import sys
# Parameters
# ==================================================

input_file=sys.argv[1] # 待评估文件
eval_file= '{}_format'.format(input_file) # 待评估文件格式化后文件
output_file=sys.argv[2] # 评估后输出文件


print (input_file)
print (output_file)

# Eval Parameters
batch_size=64 # 批大小
vocab_filepath='./vocab/vocab'#训练使使用的词表
model='./models/model-11000' #加载训练模型
allow_soft_placement=True
log_device_placement=False
confidence=0.8

if eval_file==None or vocab_filepath==None or model==None :
    print("Eval or Vocab filepaths are empty.")
    exit()

# load data and map id-transform based on training time vocabulary
inpH = InputHelper()
inpH.train_file_preprocess(input_file, eval_file)
x1_test,x2_test,y_test = inpH.getTestDataSet(eval_file, vocab_filepath, 30)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = model
print checkpoint_file
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=allow_soft_placement,
      log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
        input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/distance").outputs[0]

        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        sim = graph.get_operation_by_name("accuracy/temp_sim").outputs[0]

        #emb = graph.get_operation_by_name("embedding/W").outputs[0]
        #embedded_chars = tf.nn.embedding_lookup(emb,input_x)
        # Generate batches for one epoch
        batches = inpH.batch_iter(list(zip(x1_test,x2_test,y_test)), 2 * batch_size, 1, shuffle=False)
        # Collect the predictions here
        all_predictions = []
        all_d=[]

        for db in batches:
            x1_dev_b,x2_dev_b,y_dev_b = zip(*db)
            batch_predictions, batch_acc, batch_sim = sess.run([predictions,accuracy,sim], {input_x1: x1_dev_b, input_x2: x2_dev_b, input_y:y_dev_b, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            print(batch_predictions)
            all_d = np.concatenate([all_d, batch_sim])
            print("DEV acc {}".format(batch_acc))
        for ex in all_predictions:
            print ex

        f_output = open(output_file, 'a')
        index = 1
        for item in all_d:
            f_output.write('{}\t{}\n'.format(index, item))
            index+=1

        correct_predictions = float(np.mean(all_d == y_test))
        print("Accuracy: {:g}".format(correct_predictions))


