import torch
import torch.nn as nn
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
import math
import spacy
import argparse

from models import Encoder
from models import Decoder
from models import Seq2seq
import train
import random
import time
import math

"""
设置随机种子
import random
import time
import math
# random.seed()没有参数时，每次生成的随机数是不一样的
# 而当seed()有参数时，每次生成的随机数是一样的
# 同时选择不同的参数生成的随机数也不一样

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
# 设置这个 flag 为 True，我们就可以在 PyTorch 中对模型里的卷积层进行预先的优化，也就是在每一个卷积层中测试 cuDNN 提供的所有卷积实现算法，然后选择最快的那个
torch.backends.cudann.deterministic = True
"""


def parser_set():
    parser = argparse.ArgumentParser(description='机器翻译')
    parser.add_argument('-train', action='store_true', default=False, help='进行模型训练')
    parser.add_argument('-test', action='store_true', default=False, help='是否测试')
    args = parser.parse_args()

    # 模型相关
    args.enc_embed_dim = 300
    args.dec_embed_dim = 300
    args.hidden_dim = 512
    args.num_layers = 2
    args.enc_dropout = 0.5
    args.dec_dropout = 0.5
    args.teacher_force_rat = 0.5
    # 训练相关
    args.save_dir = './best'
    args.lr = 0.001
    args.clip = 1
    args.epochs = 10
    args.batch_size = 64
    # SEED = 1234
    # random.seed(SEED)
    # torch.manual_seed(SEED)
    return args


def tokenize_de(text):
    # 进行分词 并颠倒顺序
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]


def tokenize_en(text):
    # 进行分词 不颠倒顺序
    return [tok.text for tok in spacy_en.tokenizer(text)]


def define_dict():
    args.SRC_Field = Field(tokenize=tokenize_de,
                           init_token='<soc>',
                           eos_token='<eos>',
                           lower=True)
    args.TRG_Field = Field(tokenize=tokenize_en,
                           init_token='<soc>',
                           eos_token='<eos>',
                           lower=True)
    print("定义字段完成")


def update_parser_set():
    args.input_dim = len(args.SRC_Field.vocab)
    args.output_dim = len(args.TRG_Field.vocab)
    args.cross_loss = nn.CrossEntropyLoss(ignore_index=args.TRG_Field.vocab.stoi['<pad>'])


def load_dataset(SRC_field, TRG_field):
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                        fields=(SRC_field, TRG_field))
    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")
    SRC_field.build_vocab(train_data, min_freq=2)
    TRG_field.build_vocab(train_data, min_freq=2)

    args.device = torch.device('cpu')
    # BucketIterator和Iterator的区别是，BucketIterator尽可能的把长度相似的句子放在一个batch里面
    # BucketIterator 会自动将输入序列进行 shuffle 并做 bucket
    # shuffle 尽量将相同的句子放在一起
    # bucket 自动将短的句子补全
    train_iter, dev_iter, test_iter = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=args.batch_size,
        device=args.device)
    # 查看batch
    batch = next(iter(train_iter))
    print(batch)
    return train_iter, dev_iter, test_iter


def init_weights(m):
    """
    初始化 权重
    :param m:
    :return:
    """
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, +0.08)


def load_model(model):
    print("加载模型")
    f = open("best_model.txt", 'r')
    args.snapshot = f.readline()
    f.close()
    model.load_state_dict(torch.load(args.snapshot))


# 分别下载英语和德语的模型
# python -m spacy download en
# python -m spacy download de


if __name__ == '__main__':
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')
    args = parser_set()
    define_dict()
    train_iter, dev_iter, test_iter = load_dataset(args.SRC_Field, args.TRG_Field)
    update_parser_set()

    print("定义各个模型")
    args.encoder = Encoder(args)
    args.decoder = Decoder(args)
    print("\n 参数设置：")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    model = Seq2seq(args)
    if args.test:
        try:
            load_model(model)
            print("开始测试模型")
            test_loss = train.eval(test_iter, model, args.cross_loss)
            print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
        except Exception as e:
            print("测试集不存在 测试错误.")
    elif args.train:
        print("开始训练模型")
        model.apply(init_weights)
        train.train_model(train_iter, dev_iter, model, args)
        print("训练完成")
    else:
        print("请输入操作 python main.py -train | -test")




