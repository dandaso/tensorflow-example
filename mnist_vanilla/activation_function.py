import numpy as np
# 活性化関数
class ActivationFunction:
    @classmethod
    def sigmoid(cls, x):
        return 1 / (1 + np.exp(-x))

    @classmethod
    def sigmoid_grad(cls, x):
        return (1.0 - ActivationFunction.sigmoid(x)) * ActivationFunction.sigmoid(x)
 
    @classmethod
    def relu(cla, a):
        return np.maximum(0, a)

    @classmethod
    def step(cls, a):
        return np.array(x > 0, dtype=np.int)

if __name__ == '__main__':
    print(1)
