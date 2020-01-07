import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import models
import train_data
import dataset


def parser_set():
    # 命令行参数解析
    # 输入的参数 都会对参数进行数据处理
    # 如 python main.py -lr 0.01 就可以设置学习率为0.01
    parser = argparse.ArgumentParser(description='CNN 句子分类器')

    # 训练相关参数
    parser.add_argument('-lr', type=float, default=0.001, help='初始化学习率[默认: 0.001]')
    parser.add_argument('-epochs', type=int, default=128, help='训练中 总的数据训练轮数[默认: 128]')
    parser.add_argument('-batch-size', type=int, default=64, help='训练中 一个批量的数据个数[默认: 64]')
    parser.add_argument('-log-interval', type=int, default=1, help='多少次迭代进行训练打印[default: 1]')
    parser.add_argument('-test-interval', type=int, default=100, help='多少次迭代进行测试[default: 100]')
    parser.add_argument('-save-interval', type=int, default=500, help='多少次迭代进行保存数据[default:500]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='快照保存位置')
    parser.add_argument('-early-stop', type=int, default=1000, help='效果没有增加 则 多少次迭代停止')
    parser.add_argument('-save-best', type=bool, default=True, help='最好效果时 是否保存数据')
    # data
    parser.add_argument('-shuffle', action='store_true', default=False, help='每轮 是否都进行随机数据')
    # model
    parser.add_argument('-dropout', type=float, default=0.5, help='随机丢弃的概率[默认: 0.5]')
    parser.add_argument('-max-norm', type=float, default=3.0, help='L2正则化[默认: 3.0]')
    parser.add_argument('-embed-dim', type=int, default=128, help='词向量维度[default: 128]')
    parser.add_argument('-kernel-num', type=int, default=100, help='每种卷积核的个数')
    parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='卷积核的大小 三种')
    parser.add_argument('-static', action='store_true', default=False, help='是否修改词向量')
    # device
    parser.add_argument('-device', type=int, default=-1, help='进行数据迭代的设别 -1表示CPU[默认: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='不使用GPU')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='模型快照的文件名[默认: None]')
    # 当在终端运行的时候，如果不加入 -train, 那么程序running的时候，train的值为默认值: False
    # 如果加上了-train,不需要指定True/False,那么程序running的时候，train的值为True
    parser.add_argument('-train', action='store_true', default=False, help='进行模型训练')
    parser.add_argument('-test', action='store_true', default=False, help='是否测试')
    parser.add_argument('-predict', type=str, default=None, help='预测给定的句子')
    args = parser.parse_args()
    return args


# 加载 SST 数据集
def sst(text_field, label_field, **kwargs):
    # 采用 torchtext库中的 数据集用于SST
    train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
    # 构建词表，即需要给每个单词编码，也就是用数字表示每个单词，这样才能传入模型
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train_data, dev_data, test_data),
        batch_sizes=(args.batch_size, len(dev_data), len(test_data)),
        **kwargs)
    return train_iter, dev_iter, test_iter


# 加载 MR 数据集
def mr(text_field, label_field, **kwargs):
    # 采用自己写的 dataset 数据集用于MR
    # train_data 9596个 len(train_data)
    # dev_data 1066个 len(dev_data)
    train_data, dev_data = dataset.MR.splits(text_field, label_field)
    # 构建词表，即需要给每个单词编码，也就是用数字表示每个单词，这样才能传入模型
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    # 创建一个迭代器 从数据集中 加载 batches 的数据
    # 为数据集的多个拆分 创建迭代器 生成 batch
    train_iter, dev_iter = data.Iterator.splits(
        (train_data, dev_data),
        batch_sizes=(args.batch_size, len(dev_data)),
        **kwargs)
    # train_iter batch 150个 9596/64=150
    # dev_iter batch 1个
    print("train_iter 一共分为这么多个batch： {}".format(len(train_iter)))
    print("dev_iter 一共分为这么多个batch： {}".format(len(dev_iter)))
    return train_iter, dev_iter


# 主程序启动的地方
if __name__ == '__main__':
    # 对参数进行设置
    args = parser_set()
    # 加载数据集
    print("正在加载数据集...")
    # 首先定义字段的处理方法 包含一写文本处理的通用参数的设置
    # lower 是否把数据转化为小写
    # sequential 是否把数据表示成序列，如果是False, 不能使用分词
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    train_iter, dev_iter = mr(text_field, label_field, device=-1,
                              repeat=False)  # 加载MR的数据集 传入 样本的field和标签的field格式， 设备选择CPU

    # 更新参数 并打印
    args.embed_num = len(text_field.vocab)
    args.class_num = len(label_field.vocab)
    args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
    # 将kernel_sizes 划分出来
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    print("\n 参数设置：")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # 构建模型实例
    cnn = models.CNNText(args)
    # 存在快照 就加载快照
    if args.snapshot is not None:
        print("\n加载模型从 {}...".format(args.snapshot))
        # 加载快照
        cnn.load_state_dict(torch.load(args.snapshot))

    # 如果有 cuda 就调用 cuda 来跑模型
    if args.cuda:
        torch.cuda.set_device(args.device)
        cnn = cnn.cuda()

    # 训练和测试模型
    # 预测命令存在 则进行预测
    if args.predict is not None:
        print("开始预测模型")
        label = train_data.predict(args.predict, cnn, text_field, label_field, False)
        print('\n[Text]  {}\n[label]  {}\n'.format(args.predict, label))
    # 存在测试命令 就进行测试
    elif args.test:
        try:
            train_data.eval(test_iter, cnn, args)
        except Exception as e:
            print("\n测试集不存在.\n")
    # 否则就进行训练
    elif args.train:
        print("开始训练模型")
        try:
            train_data.train(train_iter, dev_iter, cnn, args)
        except KeyboardInterrupt:
            print('\n' + '-' * 89)
            print('退出了')
    else:
        print("请输入 需要执行的操作 -train -test -predict=str 等等")