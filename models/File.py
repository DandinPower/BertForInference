import json
import multiprocessing
import os
import torch
from torch import nn
import pandas as pd # 引用套件並縮寫為 pd  
from models.BertModel import *

def load_pretrained_model(pretrained_dir, num_hiddens, ffn_num_hiddens,num_heads, num_layers, dropout, max_len, devices):
    data_dir = pretrained_dir
    data_dir = 'data/bert.small.torch/'
    #讀取vocab
    vocab = Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir,
        'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    #加載預訓練模組
    bert = BERTModel(vocab_size = len(vocab), num_hiddens = num_hiddens,ffn_num_hiddens = ffn_num_hiddens, num_heads = num_heads,num_layers = num_layers, dropout = dropout, max_len = max_len,norm_shape = [256],ffn_num_input = 256)
    bert.load_state_dict(torch.load(os.path.join(data_dir,
                                                 'pretrained.params')))
    return bert, vocab

def load_finetune_model(path, num_hiddens, ffn_num_hiddens,num_heads, num_layers, dropout, max_len, devices):
    data_dir = 'data/bert.small.torch/'
    #讀取vocab
    vocab = Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir,
        'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    model = BERTModel(vocab_size = len(vocab), num_hiddens = num_hiddens,ffn_num_hiddens = ffn_num_hiddens, num_heads = num_heads,num_layers = num_layers, dropout = dropout, max_len = max_len,norm_shape = [256],ffn_num_input = 256)
    bert = BERTClassifier(model)
    bert.load_state_dict(torch.load(path))
    return bert, vocab