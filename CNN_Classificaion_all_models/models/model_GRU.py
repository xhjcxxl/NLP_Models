import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np


class GRU(nn.Module):
    """
    自定义双通道 GRU 模型
    一个通道用来 随机embedding
    一个通道用来 预训练embedding
    """

    def __init__(self, args):
        super(GRU, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num  # 词的个数
        D = args.embed_dim  # 词的维度（一个词的向量）
        C = args.class_num  # 多少个分类

        # nn.Embedding 设置词向量，生成随机矩阵，V行，D列，V*D
        self.embed = nn.Embedding(V, D)
        # 如果使用预训练embedding 想要微调权重 就需要下面这句
        # self.embed.weight.requires_grad=True

        # GRU
        #     input(seq_len, batch, input_size)
        #         input包含了 这三个参数，但是喂数据给模型的时候是包括了这些数据的，所以不用管
        #         所以 这里只输入了 D 词的维度，因为 我们输入的数据是 是包括完整的数据的 是一个 feature
        # hidden_size：隐藏层的特征维度
        # num_layers：lstm隐层的层数，默认为1
        # bias：False则bih=0和bhh=0. 默认为True
        # batch_first：True则输入输出的数据格式为 (batch, seq, feature)
        # dropout：除最后一层，每一层的输出都进行dropout，默认为: 0
        self.gru = nn.GRU(D, self.hidden_dim, dropout=args.dropout, num_layers=self.num_layers)
        # 进行dropout处理 避免过拟合
        # 定义dropout
        self.dropout = nn.Dropout(args.dropout)
        # 线性层
        self.hidden2lable = nn.Linear(self.hidden_dim, C)

    # 重载 前向传播  自定义前向传播
    def forward(self, x):
        # 前向传播 通过模型计算预测值
        embed = self.embed(x)  # (N, W, D)
        x = embed.view(len(x), embed.size(1), -1)

        gru_out, _ = self.gru(x)
        # lstm_out, self.hidden = self.lstm(x, self.hidden)
        # 返回输入矩阵input的转置，交换维度dim0和dim1。输入张量与输出张量共享内存
        gru_out = torch.transpose(gru_out, 0, 1)
        gru_out = torch.transpose(gru_out, 1, 2)

        # 激活函数
        gru_out = F.tanh(gru_out)
        # 池化
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)

        gru_out = self.dropout(gru_out)
        # 线性层
        logit = self.hidden2lable(gru_out)
        return logit
