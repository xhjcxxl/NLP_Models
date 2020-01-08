import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNNText(nn.Module):
    """
    自定义一个 CNN 模型
    """

    def __init__(self, args):
        super(CNNText, self).__init__()
        self.args = args
        V = args.embed_num  # 词的个数
        D = args.embed_dim  # 词的维度（一个词的向量）
        C = args.class_num  # 多少个分类
        Ci = 1  # 通道数 输入的维度
        Co = args.kernel_num  # 卷积核的个数（多个卷积）
        Ks = args.kernel_sizes  # 卷积的尺寸 list[3,4,5]

        # nn.Embedding 词向量化，生成随机矩阵，V行，D列，V*D
        self.embed = nn.Embedding(V, D)
        # 创建卷积层 创建一个二维卷积
        # 这里为什么要用K D呢，因为 在NLP中，只有上下滑动，没有左右滑动
        # 所以 卷积核的 列数就是固定为 词的维度
        # Ci 为输入维度 Co 为输出维度
        # ModuleList是将子Module作为一个List来保存的，可以通过下标进行访问
        # 定义一层卷积层
        # 因为 Ks有多个则有多个卷积核， 3*D，4*D，5*D，故 有三个卷积层使用各自的卷积核 分别同时进行运算
        # Co = 100 表示 一个卷积层 会使用 一种卷积核100个，输出通道为100维
        # 一共三种卷积核 总共300个通道，这也是为什么后面要把 len(Ks) * Co 乘起来的原因
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        # 进行dropout处理 避免过拟合
        # 定义dropout
        self.dropout = nn.Dropout(args.dropout)
        # 线性处理 全连接层
        # 定义全连接层 C = A*(len(Ks)*Co) 2 = A*300
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        """
        先卷积 然后 使用激活函数 然后使用 最大池化
        :param x:
        :param conv:
        :return:
        """
        x = F.relu(conv(x)).squeeze(3)   # 激活函数 并压缩维度 在第三维度上压缩一个维度
        x = F.max_pool1d(x, x.size(2)).squeeze(2)   # 最大池化 选择最大那个 在第二维度上压缩一个维度
        return x

    # 重载 前向传播  自定义前向传播
    def forward(self, x):
        # 前向传播 通过模型计算预测值
        x = self.embed(x)  # (N, W, D)
        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # 先将数据 在第一维度上扩张一个维度 (N, Ci, W, D)
        # 卷积层 + 激活函数
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        # 池化层
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        # 按列拼接数据
        x = torch.cat(x, 1)
        # 随机丢弃处理
        x = self.dropout(x)
        # 全连接层
        logit = self.fc1(x)
        return logit
