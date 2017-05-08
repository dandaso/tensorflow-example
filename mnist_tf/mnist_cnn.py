# -*- coding: utf-8 -*-
import tensorflow as tf
import time

mnist = tf.contrib.learn.datasets.mnist.read_data_sets("MNIST_data/", one_hot=True)

