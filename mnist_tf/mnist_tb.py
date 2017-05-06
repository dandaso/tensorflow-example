# -*- coding: utf-8 -*-
import tensorflow as tf
import time

#https://www.tensorflow.org/get_started/mnist/beginners
# 開始時刻
start_time = time.time()
print("start: " + str(start_time))

# MNISTデータの読み込み
# 60000点の訓練データ（mnist.train）と10000点のテストデータ（mnist.test）がある
# 訓練データとテストデータにはそれぞれ0-9の画像とそれに対応するラベル（0-9）がある
# 画像は28x28px(=784)のサイズ
# mnist.train.imagesは[60000, 784]の配列であり、mnist.train.lablesは[60000, 10]の配列
# lablesの配列は、対応するimagesの画像が3の数字であるならば、[0,0,0,1,0,0,0,0,0,0]となっている
# mnist.test.imagesは[10000, 784]の配列であり、mnist.test.lablesは[10000, 10]の配列
print("--- MNISTデータの読み込み開始 ---")
mnist = tf.contrib.learn.datasets.mnist.read_data_sets("MNIST_data/", one_hot=True)
print("--- MNISTデータの読み込み完了 ---")

# 訓練画像を入れる変数
# 訓練画像は28x28pxであり、これらを1行784列のベクトルに並び替え格納する
# Noneとなっているのは訓練画像がいくつでも入れられるようにするため
with tf.name_scope("input_x") as scope:
    x = tf.placeholder(tf.float32, [None, 784], name="input")

# 重み
# 訓練画像のpx数の行、ラベル（0-9の数字の個数）数の列の行列
# 初期値として0を入れておく
# input 784次元に対して、output 10次元としている

W = tf.Variable(tf.zeros([784, 10]), name="wight")

# バイアス
# ラベル数の列の行列
# 初期値として0を入れておく
# Wに対応するbは加算するためWのoutputと同じでなければならない
b = tf.Variable(tf.zeros([10]), name="byas")

# ソフトマックス回帰を実行
# yは入力x（画像）に対しそれがある数字である確率の分布
# matmul関数で行列xとWの掛け算を行った後、bを加算する。
# yは[1, 10]の行列
with tf.name_scope("output_y") as scope:
    y = tf.nn.softmax(tf.matmul(x, W) + b)

# y_は正解データのラベル
with tf.name_scope("label_y_") as scope:
    y_ = tf.placeholder(tf.float32, [None, 10], "y_")

# 交差エントロピー
with tf.name_scope("cross_entropy") as scope:
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    # Tensorboardのscalarタブに cross_entropyを追加
    cross_entropy_summary = tf.summary.scalar("cross_entropy", cross_entropy)

# 勾配硬化法を用い交差エントロピーが最小となるようyを最適化する
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


with tf.name_scope("accuracy") as scope:
# 正しいかの予測
# 計算された画像がどの数字であるかの予測yと正解ラベルy_を比較する
# 同じ値であればTrueが返される
# argmaxは配列の中で一番値の大きい箇所のindexが返される
# 一番値が大きいindexということは、それがその数字である確率が一番大きいということ
# Trueが返ってくるということは訓練した結果と回答が同じということ
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# 精度の計算
# correct_predictionはbooleanなのでfloatにキャストし、平均値を計算する
# Trueならば1、Falseならば0に変換される
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# Tensorboardのscalarタブに accuracyを追加
    accuracy_summary = tf.summary.scalar("accuracy", accuracy)

# 1000回の訓練（train_step）を実行する
# next_batch(100)で100つのランダムな訓練セット（画像と対応するラベル）を選択する
# 訓練データは60000点あるので全て使いたいところだが費用つまり時間がかかるのでランダムな100つを使う
# 100つでも同じような結果を得ることができる
# feed_dictでplaceholderに値を入力することができる
print("--- 訓練開始 ---")
# Sessionを開始する
with tf.Session() as sess:
    # サマリーの設定し、すべてのsummariesをmerge
    w_hist = tf.summary.histogram("weights", W)
    b_hist = tf.summary.histogram("biases", b)
    y_hist = tf.summary.histogram("y", y)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("MNIST_data", sess.graph) 
    # 初期化
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        batch = mnist.train.next_batch(100)
        if i % 10 == 0:
            summary = sess.run(merged,feed_dict={x: batch[0], y_: batch[1]})
            writer.add_summary(summary, i) 
        else:
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

    # 精度の実行と表示
    # テストデータの画像とラベルで精度を確認する
    # ソフトマックス回帰によってWとbの値が計算されているので、xを入力することでyが計算できる
    print("--- 訓練終了 ---")
    print("精度")
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# 終了時刻
end_time = time.time()
print("終了時刻: " + str(end_time))
print("かかった時間: " + str(end_time - start_time))
