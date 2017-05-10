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

## プログラム

- 1.mnist.py  
 もっとも単純なニューラルネットワーク、各関数をバラして、わかりやすくしてある
- 2.mnist_step.py  
 mnist.pyを改良し、学習の途中経過などが見れるようにしてある。このプログラムがベースになる  
- 3.mnist_tb.py  
 mnist_step.pyに可視化ツール TensorBoardを導入して見やすくした  
- 4.mnist_mlp.py  
 mnist_step.pyを複数レイヤーにしたバージョン  
- 5.mnist_cnn.py  
 mnist_mnn.pyを参考に畳み込みネットワークを実装したバージョン  
- 6.mnist_multi_optimizer.py  
 mnist_cnn.pyを勾配降下法以外の、AdaGrad、	Momentum、Adamなど様々な方法で実装

## TensorBoard
```
tensorboard --logdir=`pwd`/MNIST_data
```
起動後に http://localhost:6006/ にアクセス

## 参考URL

### 構築
【Mac】Python3（Anaconda）でTensorFlow環境を構築してみる  
http://no-title.com/programming/python3-tensorflow  

### MNIST
初心者のためのMNIST  
https://www.tensorflow.org/get_started/mnist/beginners  
TensorFlow本家のチュートリアル  
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist  
TensorFlow MNIST For ML Beginners チュートリアルの実施  
http://qiita.com/uramonk/items/c207c948ccb6cd0a1346  
TensorFlowチュートリアル - TensorFlowメカニクス101（翻訳）  
http://qiita.com/KojiOhki/items/0640d01029371d6ae092    
TensorFlowのHighLevelAPIについての雑感つき解説  
http://qiita.com/rindai87/items/72651c702e9265595047  
TensorFlowのTutorialをざっくり日本語訳していく
http://qiita.com/qooa/items/3719fec3cfe764674fb9  
TensorFlow Tutorial and Examples for beginners  
https://github.com/aymericdamien/TensorFlow-Examples/tree/master/examples  

### TensorBoard
本家サンプル  
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py  
MNIST For ML Beginnersの学習経過をTensorBoardで視覚化  
http://yoitaka.hatenablog.jp/entry/2016/12/29/000554  

### CNN
TensorflowでCNNを作る際に使いそうな関数を列挙してみた　　
http://qiita.com/tadOne/items/b484ce9f973a9f80036e　　　
