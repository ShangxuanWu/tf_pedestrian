import tensorflow as tf
import os
import pdb

input_file = '/mnt/sdc1/shangxuan/train_part0.tfrecords'
 
for serialized_example in tf.python_io.tf_record_iterator(input_file):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
     
    feature = example.features.feature["image_raw"]
    pdb.set_trace()
    label = example.features.feature["label"].float_list.value
 
    print("Feature: {}, label: {}".format(feature, label))
