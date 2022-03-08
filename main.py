import json
import multiprocessing
import os
import torch
from torch import nn
import pandas as pd # 引用套件並縮寫為 pd  
from models.BertModel import *
from models.File import load_pretrained_model,load_finetune_model

def main():
    devices = d2l.try_all_gpus()
    batch_size, max_len= 32, 512
    train_test_rate = 0.9
    lr, num_epochs = 1e-4, 5
    model_save_path = "models/bert_finetune.model"
    dataset_path = 'dataset/reviews_medium.csv'
    print("Load finetune model...")
    bert, vocab = load_finetune_model(model_save_path, num_hiddens=256, ffn_num_hiddens=512, num_heads=4,num_layers=2, dropout=0.1, max_len=512, devices=devices)
    net = bert
    testDataset = YelpDataset(dataset_path, max_len, vocab, False, 0.5)
    test_iter = torch.utils.data.DataLoader(testDataset, batch_size)
    print("testing")
    test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
    print(test_acc)
    print(f'test acc {test_acc:.3f}')
if __name__ == "__main__":
    main()