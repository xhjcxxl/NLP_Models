import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_LSTM(nn.Module):
    """
    自定义一个 CNN 模型
    """

    def __init__(self, args):
        super(CNN_LSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.lstm_num_layers = args.lstm_num_layers
        V = args.embed_num  # 词的个数
        D = args.embed_dim  # 词的维度（一个词的向量）
        C = args.class_num  # 多少个分类
        Ci = 1  # 通道数 输入的维度
        Co = args.kernel_num  # 卷积核的个数（多个卷积）
        Ks = args.kernel_sizes  # 卷积的尺寸 list[3,4,5]

        # nn.Embedding 设置词向量，生成随机矩阵，V行，D列，V*D
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
        self.lstm = nn.LSTM(D, self.hidden_dim, dropout=args.dropout, num_layers=self.lstm_num_layers)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        # 进行dropout处理 避免过拟合
        # 定义dropout
        self.dropout = nn.Dropout(args.dropout)
        # 线性处理 全连接层
        # 定义全连接层
        # L = 300 + 300
        L = len(Ks) * Co + self.hidden_dim
        self.fc1 = nn.Linear(L, L // 2)
        self.fc2 = nn.Linear(L // 2, C)

    # 重载 前向传播  自定义前向传播
    def forward(self, x):
        # 前向传播 通过模型计算预测值
        embed = self.embed(x)  # (N, W, D)
        cnn_x = embed
        if self.args.static:
            cnn_x = Variable(cnn_x)

        # CNN 和 LSTM 各自单独计算， 最后合并 然后全连接
        # CNN
        cnn_x = torch.transpose(cnn_x, 0, 1)
        cnn_x = cnn_x.unsqueeze(1)  # 先将数据 在第一维度上扩张一个维度 (N, Ci, W, D)
        # 卷积层 + 激活函数
        cnn_x = [F.relu(conv(cnn_x)).squeeze(3) for conv in self.convs1]
        # 池化层
        cnn_x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_x]
        # 卷积核3*d 一共词向量 10662个词向量，因此 3*d 得到列向量 3554*1，一共100个通道
        # 卷积核4*d 一共词向量 10662个词向量，因此 4*d 得到列向量 2666*1，一共100个通道
        # 卷积核5*d 一共词向量 10662个词向量，因此 5*d 得到列向量 2133*1，一共100个通道
        # 最大池化 每个向量选出最大值 100个通道选出 100个
        # 一共300个通道一共300个[100,100,100]
        # 然后 按列 拼接起来 得到 100*3 矩阵
        cnn_x = torch.cat(cnn_x, 1)
        # 随机丢弃处理
        cnn_x = self.dropout(cnn_x)
        # LSTM
        # view()函数作用是将一个多行的Tensor,拼接成一行
        lstm_x = embed.view(len(x), embed.size(1), -1)
        lstm_out, _ = self.lstm(lstm_x)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        lstm_out = self.dropout(lstm_out)
        # LSTM cat
        cnn_x = torch.transpose(cnn_x, 0, 1)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        cnn_lstm_out = torch.cat((cnn_x, lstm_out), 0)
        cnn_lstm_out = torch.transpose(cnn_lstm_out, 0, 1)
        # 全连接层
        cnn_lstm_out = self.fc1(F.tanh(cnn_lstm_out))
        cnn_lstm_out = self.fc2(F.tanh(cnn_lstm_out))
        logit = cnn_lstm_out
        return logit
