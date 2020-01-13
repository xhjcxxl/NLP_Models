import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import numpy as np

def load_Glove_vector():
    # use torchtext to load data, no need to download dataset
    print("loading  dataset")
    text = data.Field(lower=True, include_lengths=True, batch_first=True)  # set up fields
    label = data.Field(sequential=False)
    train, test = datasets.IMDB.splits(text, label)  # make splits for data
    text.build_vocab(train, vectors=GloVe(name='6B', dim=300))  # build the vocabulary
    label.build_vocab(train)
    print('len(TEXT.vocab)', len(text.vocab))  # print vocab information
    print('TEXT.vocab.vectors.size()', text.vocab.vectors.size())

if __name__=='__main__':
    load_Glove_vector()