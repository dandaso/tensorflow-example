import tensorflow as tf
from two_layer_net import TwoLayerNet

mnist = tf.contrib.learn.datasets.mnist.read_data_sets("MNIST_data/", one_hot=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

epoch = 10000
batch_size = 200
learning_rate = 0.1

for i in range(epoch):
    batch = mnist.train.next_batch(100)
    #grad = network.numerical_gradient(batch[0], batch[1])
    grad = network.gradient(batch[0], batch[1])
    for key in("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    if i % 100 == 0:
        acc = network.accuracy(mnist.test.images, mnist.test.labels)
        print("accuracy: {0}".format(acc))
