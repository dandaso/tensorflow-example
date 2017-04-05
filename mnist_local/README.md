## 環境設定

- Anacondaをインストールする
```
python --version
Python 3.5.2 :: Anaconda 4.2.0 (x86_64)
```
- Anacondaの仮想環境tf内にTensorFlowをインストールする  
```
conda create -n tf python=3.5.2
source $PYENV_ROOT/versions/anaconda3-4.2.0/bin/activate
conda install -c conda-forge tensorflow
python -c "import tensorflow as tf; print(tf.__version__)"
1.0.0
```


## TensorFlowの大まかな流れ
- 1.入力用にplaceholderを用意
- 2.重みやバイアス用にVariableを用意
- 3.活性化関数を使って層を定義
- 4.ロス値を定義
- 5.最適化の方法を定義
- 6.セッションを定義して学習を実行

## 実行


## 参考URL

http://no-title.com/programming/python3-tensorflow  
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist
http://qiita.com/uramonk/items/c207c948ccb6cd0a1346  
http://qiita.com/KojiOhki/items/0640d01029371d6ae092  
http://qiita.com/rindai87/items/72651c702e9265595047  
