import numpy as np

class Output:
    @classmethod  
    def softmax(cls, a):
        c = np.max(a)
        # 値がマイナスにならないようにexp関数を用いる
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        return exp_a / sum_exp_a

if __name__ == '__main__':
    a = np.array([0.3, 2.9, 4.0])
    y = Output.softmax(a)
    print(y)
    print(np.sum(y))
