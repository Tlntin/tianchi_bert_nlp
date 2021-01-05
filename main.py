import torch
from torch import nn
from BertRCNN.BertRCNN import BertRCNN
import pandas as pd
import os
from utils.train_valid import train_valid
from config import Config
from utils.tools import set_random_seed
from utils.test import test_model
from utils.train_valid import train_enhance
from utils.epoch_loss_weight import EpochLossWeight
from utils.data_preprocess import DataPreProcess
from pytorch_pretrained_bert.optimization import BertAdam


if __name__ == '__main__':
    # 固定随机种子
    config = Config()
    if config.seed:
        set_random_seed(config)
    # 读取所有train/valid样本
    split_train_dir = os.path.join(config.train_split_dir, config.file_name_list[0])
    split_valid_dir = os.path.join(config.valid_split_dir, config.file_name_list[0])
    if len(os.listdir(split_train_dir)) == 0 or len(os.listdir(split_valid_dir)) == 0:
        print('没有发现分割数据集，开始分割')
        data_pre = DataPreProcess()
        data_pre.run()
    else:
        print('本次无需分割数据集')
        sub_length_dict = {}
        for file_name in config.file_name_list:
            split_train_dir = os.path.join(config.train_split_dir, file_name)
            file_path = os.path.join(split_train_dir, '1.csv')
            sub_df = pd.read_csv(file_path, index_col=0, encoding='utf-8')
            sub_length_dict[file_name] = len(sub_df)
        print('子集长度分别为：', sub_length_dict)
    # 正式准备训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertRCNN.from_pretrained(config.pre_model_path, rnn_hidden_size=config.hidden_size).to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    criterion = nn.CrossEntropyLoss().to(device)
    epoch_total = 0  # 单个样本的训练次数
    for file_name in config.file_name_list:
        batch_size = config.batch_size_dict[file_name]
        print(f'file_name:{file_name}, batch_size:{batch_size}')
        train_df = pd.read_csv(os.path.join(config.processed_train, file_name + '.csv'), index_col=0)
        step_grad = config.gradient_accumulation_step_dict[file_name]
        epoch_total += (len(train_df) // batch_size + 1) // step_grad + 1
    print('epoch_total', epoch_total)
    optimizer = BertAdam(optimizer_grouped_parameters, lr=config.learning_rate,
                         warmup=config.warmup_proportion, t_total=epoch_total * config.epochs)
    # 下面这行根据实际需求是否注释
    model, optimizer = train_valid(model, criterion, optimizer)
    loss_weights = EpochLossWeight()
    loss_weights_dict = loss_weights.run()
    # 进行最后一次模型数据增强与结果测试
    # if os.path.exists(config.checkpoint_file):
    #     print('正在加载最后一个模型', config.checkpoint_file)
    #     checkpoint = torch.load(config.checkpoint_file)
    #     model.load_state_dict(checkpoint['model_state'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state'])
    train_enhance(model, criterion, optimizer, loss_weights_dict)
    test_model(model, 'last_predict', 'last_predict_submit.zip')
    # 进行最佳模型的的数据增强与结果测试
    if os.path.exists(config.best_checkpoint_file):
        print('正在加载最佳模型', config.best_checkpoint_file)
        checkpoint = torch.load(config.best_checkpoint_file)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    train_enhance(model, criterion, optimizer, loss_weights_dict)
    test_model(model, 'best_predict', 'best_predict_submit.zip')

