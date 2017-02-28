# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import pdb
import scipy.io as sio
import tensorflow as tf
import cv2

from tensorflow.contrib.learn.python.learn.datasets import mnist

FLAGS = None


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_txt(file_name):
    inputLines = []
    with open(file_name) as inputStream:
        for separateLine in iter(inputStream):
            inputLines.append(separateLine.strip('\n'))
    return inputLines

def convert_to(file_list, name, i):
  """Converts a dataset to tfrecords."""
  # N W H C shape numpy array, place my things here

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for offset in range(5000):
    ii = offset + i * 5000
    img_fn = file_list[ii]
    print(ii)
    #print("&&")
    #print(offset
    # image to string
    image = cv2.imread(img_fn)
    image = (image - 128) / 128
    image_ = image[50:418,136:504,:]
    #pdb.set_trace()
    image_raw = image_.tostring()
    
    # labels to string
    label_fn = img_fn.replace('image', 'gt')[:-4]+'.mat'
    label = sio.loadmat(label_fn)['heatmap']
    label_raw = label.tostring();
    pdb.set_trace()
    
    
    example = tf.train.Example(features=tf.train.Features(feature={
        #'height': _int64_feature(rows),
        #'width': _int64_feature(cols),
        #'depth': _int64_feature(depth),
        'label': _bytes_feature(label_raw),
        'image_raw': _bytes_feature(image_raw)}))
        
    writer.write(example.SerializeToString())
  writer.close()

def main(unused_argv):
  for i in range(10):
  # Get the data.
      #data_sets = mnist.read_data_sets(FLAGS.directory,
      #                                 dtype=tf.uint8,
      #                                 reshape=False,
      #                                 validation_size=FLAGS.validation_size)

      # Convert to Examples and write the result to TFRecords.
      file_list = read_txt('/home/shangxuan/tf_pedestrian/file_lists/train_list_3.txt')
      convert_to(file_list, 'train_part'+str(i), i)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      default='/mnt/sdc1/shangxuan/synthetic3_tf',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--validation_size',
      type=int,
      default=5000,
      help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
