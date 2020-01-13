import os
import argparse
import datetime
import torch
import torchtext.data as data
import train_cnn
import train_lstm
import dataset
from torchtext.vocab import Vectors
from models.model_CNN import CNNText
from models.model_mulit_CNN import MultiCNNText
from models.model_LSTM import LSTM
from models.model_GRU import GRU
from models.model_BiLSTM import BiLSTM
from models.model_CNN_LSTM import CNN_LSTM
from models.model_BiGRU import BiGRU


cache = '.vector_cache'
def parser_set():
    # 命令行参数解析
    # 输入的参数 都会对参数进行数据处理
    # 如 python main.py -lr 0.01 就可以设置学习率为0.01
    parser = argparse.ArgumentParser(description='CNN 句子分类器')

    # 训练相关参数
    parser.add_argument('-lr', type=float, default=0.001, help='初始化学习率[默认: 0.001]')
    parser.add_argument('-epochs', type=int, default=256, help='训练中 总的数据训练轮数[默认: 256]')
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
    parser.add_argument('-embed-dim', type=int, default=300, help='Glove词向量维度[默认: 300]')
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
    parser.add_argument('-embed-type', type=str, default='rand', help='使用什么embedding，默认使用随机embedding')
    parser.add_argument('-model-type', type=str, default='CNN', help='使用什么模型，默认CNN')
    args = parser.parse_args()
    return args


def update_parser_set():
    # 词的词表大小正好就是 词向量矩阵（V*D）的行V，维度D可以自己指定
    args.embed_dropout = 0.5
    args.lstm_hidden_dim = 300
    args.lstm_num_layers = 1
    args.lstm_weight_init = True
    args.lstm_weight_init_value = 2.0
    print("update_parser_set")
    args.embed_num = len(args.text_field.vocab)
    args.class_num = len(args.label_field.vocab)
    args.cuda = (not args.no_cuda) and torch.cuda.is_available();del args.no_cuda
    # 将kernel_sizes 划分出来
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


def mr_multi(text_field, label_field, static_text_field, static_label_field, **kwargs):
    # 获取数据
    train_data, dev_data = dataset.MR.splits(text_field, label_field)
    static_train_data, static_dev_data = dataset.MR.splits(static_text_field, static_label_field)
    # 构建词表
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    # 构建 预训练词向量 词表
    if not os.path.exists(cache):
        print("不存在此目录,创建此目录")
        os.mkdir(cache)
    vectors = Vectors(name='.vector_cache/glove.6B/glove.6B.300d.txt', cache=cache)
    static_text_field.build_vocab(static_train_data, static_dev_data, vectors=vectors)
    static_label_field.build_vocab(static_train_data, static_dev_data)
    # 加载 预训练向量权重
    args.static_weight_matrix = static_text_field.vocab.vectors
    # 创建一个迭代器 从数据集中 加载 batches 的数据
    # 为数据集的多个拆分 创建迭代器 生成 batch
    train_iter, dev_iter = data.Iterator.splits(
        (train_data, dev_data),
        batch_sizes=(args.batch_size, len(dev_data)),
        **kwargs)
    return train_iter, dev_iter


# 加载 MR 数据集
def mr(text_field, label_field, **kwargs):
    # 采用自己写的 dataset 数据集用于MR
    train_data, dev_data = dataset.MR.splits(text_field, label_field)
    if args.embed_type == 'rand':
        print("使用 随机embedding")
        text_field.build_vocab(train_data, dev_data)
        label_field.build_vocab(train_data, dev_data)
    elif args.embed_type == 'static':
        print("使用 静态外部embedding")
        if not os.path.exists(cache):
            print("不存在此目录,创建此目录")
            os.mkdir(cache)
        vectors = Vectors(name='.vector_cache/glove.6B/glove.6B.300d.txt', cache=cache)
        text_field.build_vocab(train_data, dev_data, vectors=vectors)
        label_field.build_vocab(train_data, dev_data)
        args.weight_matrix = text_field.vocab.vectors
    elif args.embed_type == 'not-static':
        print("使用 非静态 外部embedding")
        if not os.path.exists(cache):
            print("不存在此目录,创建此目录")
            os.mkdir(cache)
        vectors = Vectors(name='.vector_cache/glove.6B/glove.6B.300d.txt', cache=cache)
        text_field.build_vocab(train_data, dev_data, vectors=vectors)
        label_field.build_vocab(train_data, dev_data)
        args.weight_matrix = text_field.vocab.vectors

    # 创建一个迭代器 从数据集中 加载 batches 的数据
    # 为数据集的多个拆分 创建迭代器 生成 batch
    train_iter, dev_iter = data.Iterator.splits(
        (train_data, dev_data),
        batch_sizes=(args.batch_size, len(dev_data)),
        **kwargs)
    # train_iter batch 150个 9596/64=150
    # dev_iter batch 1个
    return train_iter, dev_iter


def define_dict():
    """
    因为使用torchtext 所以使用了 定义字段
    :return:
    """
    # 首先定义字段的处理方法 包含一写文本处理的通用参数的设置
    # lower 是否把数据转化为小写
    # sequential 是否把数据表示成序列，如果是False, 不能使用分词
    args.text_field = data.Field(lower=True)
    args.label_field = data.Field(sequential=False)
    args.static_text_field = data.Field(lower=True)
    args.static_label_field = data.Field(sequential=False)
    print("定义字段完成")

def load_data(set_model, train):
    if args.model_type == 'CNN':
        print("使用CNN 模式数据集")
        # 加载MR的数据集 传入 样本的field和标签的field格式， 设备选择CPU
        train_iter, dev_iter = mr(args.text_field, args.label_field, device=-1, repeat=False)
    elif args.model_type == 'CNN_multi':
        print("使用CNN_multi 模式数据集")
        train_iter, dev_iter = mr_multi(args.text_field, args.label_field, args.static_text_field,
                                        args.static_label_field, device=-1, repeat=False)
    elif args.model_type == 'LSTM':
        print("使用lstm 模式数据集")
        train_iter, dev_iter = mr(args.text_field, args.label_field, device=-1, repeat=False)
    elif args.model_type == 'BILSTM':
        print("使用BILSTM 模式数据集")
        train_iter, dev_iter = mr(args.text_field, args.label_field, device=-1, repeat=False)
    elif args.model_type == 'GRU':
        print("使用GRU 模式数据集")
        train_iter, dev_iter = mr(args.text_field, args.label_field, device=-1, repeat=False)
    elif args.model_type == 'BIGRU':
        print("使用BIGRU 模式数据集")
        train_iter, dev_iter = mr(args.text_field, args.label_field, device=-1, repeat=False)
    elif args.model_type == 'CNN_LSTM':
        print("使用CNN_LSTM 模式数据集")
        train_iter, dev_iter = mr(args.text_field, args.label_field, device=-1, repeat=False)

    # 更新参数 并打印
    update_parser_set()
    print("\n 参数设置：")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # 构建模型实例
    if args.model_type == 'CNN':
        set_model = CNNText(args)
        train = train_cnn.train(train_iter, dev_iter, set_model, args)
    elif args.model_type == 'CNN_multi':
        set_model = MultiCNNText(args)
        train = train_cnn.train(train_iter, dev_iter, set_model, args)
    elif args.model_type == 'LSTM':
        set_model = LSTM(args)
        train = train_lstm.train(train_iter, dev_iter, set_model, args)
    elif args.model_type == 'GRU':
        set_model = GRU(args)
        train = train_lstm.train(train_iter, dev_iter, set_model, args)
    elif args.model_type == 'BILSTM':
        set_model = BiLSTM(args)
        train = train_lstm.train(train_iter, dev_iter, set_model, args)
    elif args.model_type == 'BIGRU':
        set_model = BiGRU(args)
        train = train_lstm.train(train_iter, dev_iter, set_model, args)
    elif args.model_type == 'CNN_LSTM':
        set_model = CNN_LSTM(args)
        train = train_lstm.train(train_iter, dev_iter, set_model, args)
    return train_iter, dev_iter


# 主程序启动的地方
if __name__ == '__main__':
    # 对参数进行设置
    args = parser_set()
    # 定义字段
    define_dict()
    # 加载数据集
    print("正在加载数据集...")

    set_model = None
    train = None
    train_iter, dev_iter = load_data(set_model, train)

    # 如果有 cuda 就调用 cuda 来跑模型
    if args.cuda:
        torch.cuda.set_device(args.device)
        set_model = set_model.cuda()

    # 训练和测试模型
    # 预测命令存在 则进行预测
    if args.predict is not None:
        print("加载模型")
        f = open("best_model.txt", 'r')
        args.snapshot = f.readline()
        f.close()
        print("开始预测模型")
        if args.model_type == 'lstm':
            set_model.load_state_dict(torch.load(args.snapshot))
            label = train_lstm.predict(args.predict, set_model, args.text_field, args.label_field, False)
            print('\n[Text]  {}\n[label]  {}\n'.format(args.predict, label))
        else:
            set_model.load_state_dict(torch.load(args.snapshot))
            label = train_cnn.predict(args.predict, set_model, args.text_field, args.label_field, False)
            print('\n[Text]  {}\n[label]  {}\n'.format(args.predict, label))
    # 存在测试命令 就进行测试
    elif args.test:
        try:
            train_cnn.eval(test_iter, set_model, args)
        except Exception as e:
            print("\n测试集不存在.\n")
    # 否则就进行训练
    elif args.train:
        print("开始训练模型")
        try:
            train(train_iter, dev_iter, set_model, args)
        except KeyboardInterrupt:
            print('\n' + '-' * 89)
            print('退出了')
    else:
        print("请输入 需要执行的操作 -train -test -predict=str 等等")
