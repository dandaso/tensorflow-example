import numpy as np

class LossFcuntion:
    # 二乗和誤差
    @classmethod
    def mean_squared_error(cls, y, t):
        return 0.5 * np.sum((y - t)**2)  
    # 交差エントロピー 
    @classmethod
    def cross_entropy_error(cls, y, t):
        # log0がマイナス無限大にならないように微小値を足している
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))

if __name__ == '__main__':
    # 正解ラベル
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    # 数値Nの0〜9の確率(2が一番高い場合)
    y1 = [0.1 ,0.05 ,0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]   
    loss1 = LossFcuntion.mean_squared_error(np.array(y1), np.array(t))
    loss2 = LossFcuntion.cross_entropy_error(np.array(y1), np.array(t))
    # 0.0975, 0.5108
    print(loss1, loss2)
    # 数値Nの0〜9の確率(7が一番高い場合)
    y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    loss1 = LossFcuntion.mean_squared_error(np.array(y2), np.array(t))
    loss2 = LossFcuntion.cross_entropy_error(np.array(y2), np.array(t))
    # 0.5975 2.30258
    print(loss1, loss2)
