## 训练
输入：
    多通道: python main.py -train -model-type="CNN_multi"
    单通道: python main.py -train -embed-type="static" -model-type="CNN"

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

