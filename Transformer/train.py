import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import math
import time
from model_Transformer import subsequent_mask


class Batch:
    # Object for holding a batch of data with mask during training
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(trg, pad):
        # Create a mask to hide padding and future words
        trg_mask = (trg != pad).unsqueeze(-2)
        trg_mask = trg_mask & Variable(subsequent_mask(trg.size(-1)).type_as(trg_mask.data))
        return trg_mask


def run_epoch(data_iter, model, loss_compute):
    """
    一个标准的训练epoch
    Standard Training and Logging Function
    """
    start = time.time()  # 开始时间
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        # 调用模型
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        # 计算损失
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        # 叠加损失
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f".format(i, loss/batch.ntokens, tokens/elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens
