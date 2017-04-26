# -*- coding: utf-8 -*-
import tensorflow as tf
import time

mnist = tf.contrib.learn.datasets.mnist.read_data_sets("MNIST_data/", one_hot=True)

# 入力
x = tf.placeholder(tf.float32, [None, 784], name="input")

# レイヤー1
# tf.zerosから tf.random_normal に変更になっているがなんで変わっているかは謎
# 変更しないと学習が上手くいかない
w1 = tf.Variable(tf.random_normal([784, 256]), name="wight1")
b1 = tf.Variable(tf.random_normal([256]))
h1 = tf.nn.relu(tf.matmul(x,w1) + b1)
# レイヤー2
w2 = tf.Variable(tf.random_normal([256, 256]), name="wight2")
b2 = tf.Variable(tf.random_normal([256]))
h2 = tf.nn.relu(tf.matmul(h1,w2) + b2)
# 出力レイヤー
w3 = tf.Variable(tf.random_normal([256, 10]), name="wight3")
b3 = tf.Variable(tf.random_normal([10]))
h3 = tf.matmul(h2,w3) + b3

# 結果
y_ = tf.placeholder(tf.float32, [None, 10], "y_")

# softmax関数から tf.reduce_mean(softmax_cross_entropy_with_logits) に変更になっている
# やってることは同じに見えるんだけど、なんで変更されているか謎。変更しないと学習が上手くいかない
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h3, labels=y_))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(h3,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print("--- 訓練開始 ---")
# Sessionを開始する
with tf.Session() as sess:
    # 初期化
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        batch = mnist.train.next_batch(100)
        sess.run([train_step, cross_entropy], feed_dict={x: batch[0], y_: batch[1]})
    # 精度の実行と表示
    print("--- 訓練終了 ---")
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
