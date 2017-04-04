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

## 実行


## 参考URL

http://no-title.com/programming/python3-tensorflow  
http://qiita.com/uramonk/items/c207c948ccb6cd0a1346  
http://qiita.com/KojiOhki/items/0640d01029371d6ae092  
http://qiita.com/rindai87/items/72651c702e9265595047
