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
【Mac】Python3（Anaconda）でTensorFlow環境を構築してみる  
http://no-title.com/programming/python3-tensorflow  
TensorFlow本家のチュートリアル  
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist  
TensorFlow MNIST For ML Beginners チュートリアルの実施  
http://qiita.com/uramonk/items/c207c948ccb6cd0a1346   
TensorFlowチュートリアル - TensorFlowメカニクス101（翻訳）  
http://qiita.com/KojiOhki/items/0640d01029371d6ae092    
TensorFlowのHighLevelAPIについての雑感つき解説  
http://qiita.com/rindai87/items/72651c702e9265595047  
【TensorFlowのTutorialをざっくり日本語訳していく】1. MNIST For ML Beginners  
http://qiita.com/qooa/items/3719fec3cfe764674fb9  

