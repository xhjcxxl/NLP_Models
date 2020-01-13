import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np


class BiLSTM(nn.Module):
    """
    自定义双通道 BiLSTM 模型
    一个通道用来 随机embedding
    一个通道用来 预训练embedding
    """

    def __init__(self, args):
        super(BiLSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.lstm_num_layers = args.lstm_num_layers
        V = args.embed_num  # 词的个数
        D = args.embed_dim  # 词的维度（一个词的向量）
        C = args.class_num  # 多少个分类

        # nn.Embedding 设置词向量，生成随机矩阵，V行，D列，V*D
        self.embed = nn.Embedding(V, D)
        # 如果使用预训练embedding 想要微调权重 就需要下面这句
        # self.embed.weight.requires_grad=True

        # lstm
        # 因为双向，所以隐藏层砍了一半
        self.bilstm = nn.LSTM(D, self.hidden_dim // 2, dropout=args.dropout, num_layers=self.lstm_num_layers, bidirectional=True, bias=False)
        print(self.bilstm)
        # 将lstm的所有权重矩阵进行初始化
        if args.lstm_weight_init:
            print("初始化 W 矩阵")
            init.xavier_normal(self.bilstm.all_weights[0][0], gain=np.sqrt(args.lstm_weight_init_value))
            init.xavier_normal(self.bilstm.all_weights[0][1], gain=np.sqrt(args.lstm_weight_init_value))

        # 进行dropout处理 避免过拟合
        # 定义dropout
        self.dropout = nn.Dropout(args.dropout)
        # 对 embedding进行dropout
        self.embed_dropout = nn.Dropout(args.embed_dropout)
        # 线性层
        self.hidden2lable1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2lable2 = nn.Linear(self.hidden_dim // 2, C)

    # 重载 前向传播  自定义前向传播
    def forward(self, x):
        # 前向传播 通过模型计算预测值
        embed = self.embed(x)  # (N, W, D)
        # embed = self.embed_dropout(embed)
        x = embed.view(len(x), embed.size(1), -1)

        bilstm_out, _ = self.bilstm(x)
        # bilstm_out, self.hidden = self.lstm(x, self.hidden)
        # 返回输入矩阵input的转置，交换维度dim0和dim1。输入张量与输出张量共享内存
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)

        # 池化
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)

        bilstm_out = self.dropout(bilstm_out)
        # 线性层
        bilstm_out = self.hidden2lable1(bilstm_out)
        logit = self.hidden2lable2(bilstm_out)
        return logit
