## Modified by Nicholas Carlini to match model structure for attack code.
## Original copyright license follows.


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

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile
import scipy.misc

import numpy as np
from six.moves import urllib
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

class InceptionModel:
  image_size = 300
  num_labels = 1001
  num_channels = 3
  def __init__(self, restore, sess=None):
    self.sess = sess

    self.graph = tf.Graph()
    self.graph_def = tf.GraphDef()
    with open(restore, 'rb') as f:
      self.graph_def.ParseFromString(f.read())
    with self.graph.as_default():
      tf.import_graph_def(self.graph_def)


  def predict(self, img, tanhspace=0):
    if tanhspace == 1: # if it was in tanhspace
        img = 0.5+img  # inception requires [0, 1] normalized inputs
    resized = tf.image.resize_images(img, size=[299, 299]) # inception's size
    batch_size = tf.shape(img, out_type=tf.int32)[0]

    softmax_tensor = tf.import_graph_def(
      self.graph.as_graph_def(),
      input_map={'import/input:0': tf.cast(resized, tf.float32),
        'import/InceptionV3/Predictions/Shape:0': [batch_size, self.num_labels]},
      return_elements=['import/InceptionV3/Predictions/Reshape:0'])
    return tf.reshape(softmax_tensor, [batch_size, self.num_labels])
