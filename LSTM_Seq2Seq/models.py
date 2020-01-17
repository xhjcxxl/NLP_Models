import torch
import torch.optim as optim
import torch.nn as nn
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import random


class Encoder(nn.Module):
    """
    模型 编码
    """
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        # 一个lstm内部有 四个 平常理解的 神经网络层组成的
        # 隐藏层维度 表示的是 一个lstm中 四个 神经网络层 每个神经网络层都是128维度的
        self.hidden_dim = args.hidden_dim  # 隐藏层维度可以和 词向量的维度不一样
        self.num_layers = args.num_layers
        self.input_num = args.input_dim
        self.embed_dim = args.enc_embed_dim

        self.embed = nn.Embedding(self.input_num, self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim,
                            num_layers=self.num_layers, dropout=args.enc_dropout)
        self.dropout = nn.Dropout(args.enc_dropout)

    def forward(self, text):
        embed = self.embed(text)
        embed = self.dropout(embed)
        # 输入 embed 为 句子长度 * batch_size * embedding
        # 即 一个句子的词个数，一个batch有多少个句子，词向量
        outputs, (hidden, cell) = self.lstm(embed)

        return hidden, cell


class Decoder(nn.Module):
    """
    模型解码
    """
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim  # 隐藏维度
        self.num_layers = args.num_layers  # 层数
        self.embed_dim = args.enc_embed_dim  # 词向量维度
        self.output_dim = args.output_dim  # 输出维度

        self.embed = nn.Embedding(self.output_dim, self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim,
                            num_layers=self.num_layers, dropout=args.enc_dropout)
        self.dropout = nn.Dropout(args.dec_dropout)
        self.fc1 = nn.Linear(self.hidden_dim, self.output_dim)  # 输入为隐藏维度 输出为输出维度

    def forward(self, text, hidden, cell):
        """
        在 decoder 使用了了 encoder 模型输出的 hidden和 cell 这里面包含了上下文信息
        """
        text = text.unsqueeze(0)
        embed = self.embed(text)
        embed = self.dropout(embed)
        output, (hidden, cell) = self.lstm(embed, (hidden, cell))

        logit = self.fc1(output.squeeze(0))

        return logit, hidden, cell


class Seq2seq(nn.Module):
    def __init__(self, args):
        super(Seq2seq, self).__init__()
        self.encoder = args.encoder  # encoder模型
        self.decoder = args.decoder  # decoder模型
        self.device = args.device
        assert self.encoder.hidden_dim == self.decoder.hidden_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.num_layers == self.decoder.num_layers, \
            "Number of layers  of encoder and decoder must be equal!"

    def forward(self, text, target, teacher_force_rat):
        batch_size = target.shape[1]
        max_len = target.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # 定义输出结构
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size)
        # 先 编码 encoder 获取编码结果 上下文相关信息
        # context vector可以想成是一个含有所有输入句信息的向量，也就是Encoder当中，最后一个hidden state（即 en_hidden）
        en_hidden, en_cell = self.encoder(text)
        en_input = target[0, :]

        for t in range(1, max_len):
            # 开始解码 decoder
            output, out_hidden, out_cell = self.decoder(en_input, en_hidden, en_cell)
            # 存储Decoder所有输出的张量
            outputs[t] = output
            # 该参数的作用是，当使用teacher force时，decoder网络的下一个input是目标语言的下一个字符，当不使用时，网络的下一个input是其预测出的那个字符
            # 即 是否需要 在预测过程中进行矫正，保证预测下一个字符时，前面的是正确的
            teacher_force = random.random() < teacher_force_rat
            top1 = output.max(1)[1]
            en_input = (target[t] if teacher_force else top1)

        return outputs
