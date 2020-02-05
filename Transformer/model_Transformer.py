import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import math


class EncoderDecoder(nn.Module):
    """
    重构 序列模型 高级封装
    """
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator

    def forward(self, src, trg, src_mask, trg_mask):
        # 先 编码 再 解码
        # encoder的输出作为memory 放到decoder中
        encoder_memory = self.encoder(src, src_mask)
        return self.decoder(encoder_memory, src_mask, trg, trg_mask)

    def encoder(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decoder(self, memory, src_mask, trg, trg_mask):
        return self.decoder(self.trg_embed(trg), memory, src_mask, trg_mask)


class Generator(nn.Module):
    # Define standard linear + softmax generation step
    # 定义 线性层 softmax层
    # 这个是 transformer的最后两层
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.fc = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=1)


def clones(module, N):
    """
    定义一个层复制函数，将每一层的结构执行深拷贝，并返回list形式
    因为输入时 不止一个层 所以用克隆的方法 多创造几个层出来
    层与层之间 链接用残差网络，即 有可能通过这一层 有可能跳过这一层 经典的深度学习模型方法
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """
    正则化
    定义layer的正则化类，与batchsize同的是：
    batch_size的目的是将一个epoch中的数据标准化，标准化的是不同样本中的同一特征。
    layer_size的目的是将一个样本中的数据标准化
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * mean / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    具体的sublayer 就是残差网络；定义残差连接层。
    如果直接跳过的就原样输出
    如果是经过神经网络的就正则化

    因为有多个层 有些层可以跳过的
    这里的forward函数代表的意思是：
    如果该层没有跳过的话，那么返回就是该层的计算结果
    如果该层被drop了，那么直接正则化后，把该层的输入作为输出
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class Encoder(nn.Module):
    """
    encoder 编码
    定义encoder的前向传播计算方法，重复N次
    即 前向传播 创建 N个layer层
    每个Encoder重复N = 6次encoder layer

    所以在encoder框架中
        一共有6*2=12个sublayer, 最后结果被LayerNorm之后输出
        Self-attention对应的Query, Key, Value都是x本身
        Encoder输出的结果被叫做"memory"
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    上面的层 具体的层设置
    前向传播层
    encoderlayer 包括 两个子层 一个是多头注意力机制 一个是位置信息
    每个子层后面都会跟一个正则化
    """
    def __init__(self, size, self_atten, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        # attention机制
        self.self_atten = self_atten
        # 全连接的位置信息
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # Lambda函数又称匿名函数，匿名函数就是没有名字的函数
        # 两层
        # 前面一层就是 多头注意力层
        # 后面一层就是 位置信息层 就是feed_forward
        nwe_x = self.self_atten(x, x, x, mask)
        x = self.sublayer[0](x, nwe_x)
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """
    decoder 解码
    与编码层一样 由N个层组成

    decoder框架
    每个Decoder复制了N=6份的Decoder Layer
    所有Decoder Layer对应的Memory都是一样的，来自于之前Encoder的输出
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, trg_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    具体的decoder层设置
    每一个decoder层中，由三个子层组成

    每个Decoder Layer有3层Sublayer：
        第一层进行Self-attention
        第二层进行对encoder的attention（最经典的seq2seq操作), query是上一层的x, value和key是memory
        第三层feed forward
        所以在Decoder中一共有6*3=18个sublayer.
    """
    def __init__(self, size, self_atten, src_atten, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_atten = self_atten
        self.src_atten = src_atten
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), N)

    def forward(self, x, memory, src_mask, trg_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_atten(x, x, x, trg_mask))
        x = self.sublayer[1](x, lambda x: self.src_atten(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    """
    subsequent_mask函数的作用是在解码器中保证attention为已经产生的信息，忽略未产生的信息
    """
    atten_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(atten_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    """
    一种 注意力计算的公式
    Scaled Dot-Product Attention 点积
    attention可以看做是query和key-value作用的结果，其中这三者都是向量，最后的结果可以通过加权和的方式计算，其中，权重由query和key计算得出

    输入进来的query, key, value 都是(n_batches, 1, d_k, h) 大小的tensor
    输出结果的大小为:
    Attened value: (n_batches, 1, d_k, h);
    Attention Score: (n_batches, 1, d_k, d_k)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0. -1e9)
    p_atten = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_atten = dropout(p_atten)
    return torch.matmul(p_atten, value), p_atten


class MultiHeadedAttention(nn.Module):
    """
    多头注意力机制
    输入数据包括：d_k d_v
    d_k维的query和keys
    d_v维的vuale
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        #  d_model 代表的是所有sub-layers输出结果的维度。
        #  可以理解为：输出的结果中每个维度都由不同的head组成。
        # d_model=512
        # h=8
        # d_k=64
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.atten = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 通常情况下，attention的function有两种：Dot-production和additive attention
        # 这里用的是 点积
        # 调用前面的attention
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # zip 将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        # 先对输入值x进行reshape一下，然后交换在维度1,2进行交换
        # Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [ln(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for ln, x in zip(self.linears, (query, key, value))]
        # Apply attention on all the projected vectors in batch.
        x, self.atten = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 连接 "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # size: x=（n_batches, 1, h * d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    # 每个encoder和decoder都包括一个全连接的网络层
    # Feed-Forward Networks(FFN) 就是前向传播层
    def __init__(self, d_model, d_ff, dropout=0.1):
        # d_model: attention层的输出结果的维度大小
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        new_x = self.w_1(x)
        new_x = F.relu(new_x)
        new_x = self.dropout(new_x)
        new_x = self.w_2(new_x)
        return new_x


class Embeddings(nn.Module):
    # 与其他语言模型一样，加上了softmax转化为概率
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        embed = self.embed(x)
        new_x = embed * math.sqrt(self.d_model)
        return new_x


class PositionalEncoding(nn.Module):
    """
    位置编码 向量
    由于算法中没有循环以及卷积操作，为了能体现输入数据的时序性，需要在输入时增加序列的相对位置和绝对位置。
    于是，在编码器和解码器的输入部分，增加位置的编码信息
    位置编码与输入数据有同样的维度，因此这俩个向量可以相加
    我们使用sine 和 cosine函数对位置信息进行编码。编码的范围从2π到10000×2π
    这么做的原因是对于句子中任何一个单词，PEpos+k都可以表示为PEpos的线性组合，也就是说，模型能够学习到各个单词的相对位置

    我们一般以字为单位训练transformer模型, 也就是说我们不用分词了
    """
    def __init__(self, d_model, dropout, max_len=5000):
        # Compute the positional encodings once in log space.
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # x = (1, x.size(1), d_model)
        return self.dropout(x)


# Full model
def make_model(src_vocab, trg_vocab, N=6,
               d_model=512, d_ffn=2048, h=8, dropout=0.1):
    """
    transformer整个模型，上面的都是具体的各个部分
    :param src_vocab: 输入单词的长度
    :param trg_vocab: 输出单词的长度
    :param N: encoderlayer的层数
    :param d_model: 词向量的维度
    :param d_ff: 前馈输入层
    :param h: 注意力是几头的
    :param dropout:
    :return:
    """
    c = copy.deepcopy
    atten = MultiHeadedAttention(h, d_model)  # 注意力
    ffn = PositionwiseFeedForward(d_model, d_ffn, dropout)  # 前馈输入层
    position = PositionalEncoding(d_model, dropout)  # 位置向量
    # 设定 transformer模型
    """
    编码词维度(+位置词向量)->encoder(N个encoderlayer,
        每个encoderlayer包括atten和ffn共两层，每层最后都正则化)
    编码词维度(+位置词向量)->decoder(N个decoderlayer,
        每个decoderlayer包括两个atten和ffn共三层，每层最后都正则化)
    最后输出进行generator层，输出最终结果
    """
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(atten), c(ffn), dropout), N),
        Decoder(DecoderLayer(d_model, c(atten), c(atten), c(ffn), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, trg_vocab), c(position)),
        Generator(d_model, trg_vocab))

    # 初始化参数
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
