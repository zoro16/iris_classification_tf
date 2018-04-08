from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def conv_relu(input, kernal_shape, bais_shape):
    weights = tf.get_variable("weights",
                              kernal_shape,
                              initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases",
                             bias_shape,
                             initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights, strides=[1,1,1,1], padding="SAME")
    return tf.nn.relu(conv + baises)

