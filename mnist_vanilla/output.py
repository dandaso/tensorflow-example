import numpy as np

class Output:
    @classmethod  
    def softmax(cls, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T 
        x = x - np.max(x) # オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x))

if __name__ == '__main__':
    a = np.array([0.3, 2.9, 4.0])
    y = Output.softmax(a)
    print(y)
    print(np.sum(y))
