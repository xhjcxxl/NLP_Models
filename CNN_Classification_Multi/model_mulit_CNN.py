import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MultiCNNText(nn.Module):
    """
    自定义双通道 CNN 模型
    一个通道用来 随机embedding
    一个通道用来 预训练embedding
    """

    def __init__(self, args):
        super(MultiCNNText, self).__init__()
        self.args = args
        V = args.embed_num  # 词的个数
        D = args.embed_dim  # 词的维度（一个词的向量）
        C = args.class_num  # 多少个分类
        Ci = 2  # 通道数 输入的维度 一个用于随机embedding 一个用于 预训练embedding
        Co = args.kernel_num  # 卷积核的个数（多个卷积）
        Ks = args.kernel_sizes  # 卷积的尺寸 list[3,4,5]

        # nn.Embedding 设置词向量，生成随机矩阵，V行，D列，V*D
        self.embed = nn.Embedding(V, D)
        self.static_embed = nn.Embedding(V, D)
        self.static_embed.weight.data.copy_(args.static_weight_matrix)
        # 如果想要微调权重 就需要下面这句
        # self.embed.weight.requires_grad=True

        # 创建卷积层 创建一个二维卷积
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D), bias=True) for K in Ks])
        print(self.convs1)
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        # 进行dropout处理 避免过拟合
        # 定义dropout
        self.dropout = nn.Dropout(args.dropout)

        input_feature = len(Ks) * Co
        # 线性处理 全连接层
        # a//b a除以b的商 即 取商
        # 因为 使用了双通道，故不能直接将 数据直接计算输出到2个分类
        # 故将原来的 # self.fc1 = nn.Linear(len(Ks) * Co, C) 拆分成两段，600压缩到300 300 再计算出两个分类
        self.fc1 = nn.Linear(input_feature, input_feature // 2, bias=True)
        self.fc2 = nn.Linear(input_feature // 2, C, bias=True)

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
        x_rand = self.embed(x)  # (N, W, D)
        x_static = self.static_embed(x)
        # x = torch.stack([x_static, x_no_static], 1)
        x = torch.stack([x_rand, x_static], 1)
        x = self.dropout(x)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # 卷积层 + 激活函数 #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # 池化层  [(N,Co), ...]*len(Ks)

        # 卷积核3*d 一共词向量 10662个词向量，因此 3*d 得到列向量 3554*1，一共100个通道
        # 卷积核4*d 一共词向量 10662个词向量，因此 4*d 得到列向量 2666*1，一共100个通道
        # 卷积核5*d 一共词向量 10662个词向量，因此 5*d 得到列向量 2133*1，一共100个通道
        # 最大池化 每个向量选出最大值 100个通道选出 100个
        # 一共300个通道一共300个[100,100,100]
        # 然后 按列 拼接起来 得到 100*3 矩阵
        x = torch.cat(x, 1)
        # 随机丢弃处理
        x = self.dropout(x)
        # 全连接层
        x = self.fc1(x)
        logit = self.fc2(x)
        return logit
