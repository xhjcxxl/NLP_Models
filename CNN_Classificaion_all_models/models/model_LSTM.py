import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np


class LSTM(nn.Module):
    """
    自定义双通道 LSTM 模型
    一个通道用来 随机embedding
    一个通道用来 预训练embedding
    """

    def __init__(self, args):
        super(LSTM, self).__init__()
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

        """
        input(seq_len, batch, input_size)
        input包含了 这三个参数，但是喂数据给模型的时候是包括了这些数据的，所以不用管
        所以 这里只输入了 D 词的维度，因为 我们输入的数据是 是包括完整的数据的 是一个 feature
        
        因为没有定义 h0 c0 所以这里要输入 具体的 hidden_size num_layers 然后系统自动定义
        lstm的 h0 和 c0 如果不初始化，PyTorch默认初始化为全零的张量
        h0(num_layers * num_directions, batch, hidden_size)
        c0(num_layers * num_directions, batch, hidden_size)
        
        hidden_size：隐藏层的特征维度
        num_layers：lstm隐层的层数，默认为1 双向lstm层数翻倍
        num_directions： 就是 方向，是单向lstm（1） 还是 双向lstm（2）
        bias：False则bih=0和bhh=0. 默认为True
        batch_first：True则输入输出的数据格式为 (batch, seq, feature)
        dropout：除最后一层，每一层的输出都进行dropout，默认为: 0
        """
        # lstm
        self.lstm = nn.LSTM(D, self.hidden_dim, dropout=args.dropout, num_layers=self.lstm_num_layers)
        # 将lstm的所有权重矩阵进行初始化
        # torch.nn.init.xavier_normal_(tensor, gain=1) 正态分布~N(0,std)
        if args.lstm_weight_init:
            print("初始化 W 矩阵")
            init.xavier_normal(self.lstm.all_weights[0][0], gain=np.sqrt(args.lstm_weight_init_value))
            init.xavier_normal(self.lstm.all_weights[0][1], gain=np.sqrt(args.lstm_weight_init_value))

        # 进行dropout处理 避免过拟合
        # 定义dropout
        self.dropout = nn.Dropout(args.dropout)
        # 对 embedding进行dropout
        self.embed_dropout = nn.Dropout(args.embed_dropout)
        # 线性层
        self.hidden2lable = nn.Linear(self.hidden_dim, C)

    # 重载 前向传播  自定义前向传播
    def forward(self, x):
        # 前向传播 通过模型计算预测值
        embed = self.embed(x)  # (N, W, D)
        # embed = self.embed_dropout(embed)
        x = embed.view(len(x), embed.size(1), -1)

        lstm_out, _ = self.lstm(x)
        # lstm_out, self.hidden = self.lstm(x, self.hidden)
        # 返回输入矩阵input的转置，交换维度dim0和dim1。输入张量与输出张量共享内存
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)

        # 激活函数
        lstm_out = F.tanh(lstm_out)
        # 池化
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        # dropout
        lstm_out = self.dropout(lstm_out)
        # 线性层
        logit = self.hidden2lable(lstm_out)
        return logit
