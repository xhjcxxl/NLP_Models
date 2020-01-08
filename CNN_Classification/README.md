## Introduction
This is the implementation of Kim's [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) paper in PyTorch.

1. Kim's implementation of the model in Theano:
[https://github.com/yoonkim/CNN_sentence](https://github.com/yoonkim/CNN_sentence)
2. Denny Britz has an implementation in Tensorflow:
[https://github.com/dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)
3. Alexander Rakhlin's implementation in Keras;
[https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras](https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras)

## Requirement
* python 3
* pytorch > 0.1
* torchtext > 0.1
* numpy

## Result
I just tried two dataset, MR and SST.

|Dataset|Class Size|Best Result|Kim's Paper Result|
|---|---|---|---|
|MR|2|77.5%(CNN-rand-static)|76.1%(CNN-rand-nostatic)|
|SST|5|37.2%(CNN-rand-static)|45.0%(CNN-rand-nostatic)|

I haven't adjusted the hyper-parameters for SST seriously.

## Usage
```
./main.py -h
```
or 

```
python3 main.py -h
```

You will get:

```
CNN 句子分类器

optional arguments:
  -h, --help            show this help message and exit
  -lr LR                初始化学习率[默认: 0.001]
  -epochs EPOCHS        训练中 总的数据训练轮数[默认: 128]
  -batch-size BATCH_SIZE
                        训练中 一个批量的数据个数[默认: 64]
  -log-interval LOG_INTERVAL
                        多少次迭代进行训练打印[default: 1]
  -test-interval TEST_INTERVAL
                        多少次迭代进行测试[default: 100]
  -save-interval SAVE_INTERVAL
                        多少次迭代进行保存数据[default:500]
  -save-dir SAVE_DIR    快照保存位置
  -early-stop EARLY_STOP
                        效果没有增加 则 多少次迭代停止
  -save-best SAVE_BEST  最好效果时 是否保存数据
  -shuffle              每轮 是否都进行随机数据
  -dropout DROPOUT      随机丢弃的概率[默认: 0.5]
  -max-norm MAX_NORM    L2正则化[默认: 3.0]
  -embed-dim EMBED_DIM  词向量维度[default: 128]
  -kernel-num KERNEL_NUM
                        每种卷积核的个数
  -kernel-sizes KERNEL_SIZES
                        卷积核的大小 三种
  -static               是否修改词向量
  -device DEVICE        进行数据迭代的设别 -1表示CPU[默认: -1]
  -no-cuda              不使用GPU
  -snapshot SNAPSHOT    模型快照的文件名[默认: None]
  -train                进行模型训练
  -test                 是否测试
  -predict PREDICT      预测给定的句子
```
默认参数设置：
        BATCH_SIZE=64
        CLASS_NUM=3
        CUDA=False
        DEVICE=-1
        DROPOUT=0.5
        EARLY_STOP=1000
        EMBED_DIM=128
        EMBED_NUM=21108
        EPOCHS=5
        KERNEL_NUM=100
        KERNEL_SIZES=[3, 4, 5]
        LOG_INTERVAL=1
        LR=0.001
        MAX_NORM=3.0
        PREDICT=None
        SAVE_BEST=True
        SAVE_DIR=snapshot\2020-01-08_10-21-45
        SAVE_INTERVAL=500
        SHUFFLE=False
        SNAPSHOT=None
        STATIC=False
        TEST=False
        TEST_INTERVAL=100
        TRAIN=True

## Train
```
./main.py
```
You will get:

```
Batch[100] - loss: 0.655424  acc: 59.3750%
Evaluation - loss: 0.672396  acc: 57.6923%(615/1066) 
```

## Predict
* **Example1**

	```
	./main.py -predict="Hello my dear , I love you so much ."
	```
	You will get:
	
	```
	Loading model from [./snapshot/2017-02-11_15-50-53/snapshot_steps1500.pt]...
	
	[Text]  Hello my dear , I love you so much .
	[Label] positive
	```
输入的句子 单词个数必须大于等于5个，因为卷积核的尺寸最大是5*D的，也就是一次5个词，所以必须要满足这个条件才能进行预测，这一点可以后续修改一下
Your text must be separated by space, even punctuation.And, your text should longer then the max kernel size.

## Reference
* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

