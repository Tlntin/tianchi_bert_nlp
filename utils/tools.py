import json
import os
import random

import numpy as np
import torch
from sklearn import metrics

from config import Config

config = Config()


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_score, is_best):
    """
    此函数用于保存最佳模型
    :param epoch: 第几轮训练
    :param epochs_since_improvement: 距离上次最佳模型已经经过了几轮
    :param model: 模型
    :param optimizer: 优化器
    :param best_score: 验证集最好的结果
    :param is_best: 是否最佳
    """
    model_to_save = model.module if hasattr(model, 'module') else model
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'best_score': best_score,
             'model_state': model_to_save.state_dict(),
             'optimizer_state': optimizer.state_dict()}
    torch.save(state, config.checkpoint_file)
    model_config = model_to_save.config.to_json_string()
    with open(config.checkpoint_config, 'wt', encoding='utf-8') as f:
        json.dump(model_config, f)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, config.best_checkpoint_file)
        with open(config.best_checkpoint_config, 'wt', encoding='utf-8') as f:
            json.dump(model_config, f)
        print('最佳模型已经保存模型在{}路径'.format(config.best_checkpoint_file))


def save_epoch_csv(epoch, epoch_train_loss, epoch_train_acc, epoch_train_f1_score,
                   epoch_valid_loss, epoch_valid_acc, epoch_valid_f1_score, out_path):
    """
    此文件用户储存训练过程中训练集与验证集的准确率与loss
    """
    if not os.path.exists(out_path) or epoch == 1:
        f = open(out_path, 'wt', encoding='utf-8-sig')
        columns = ['train_loss', 'train_acc', 'train_f1_score', 'valid_loss', 'valid_acc', 'valid_f1_score']
        f.write(','.join(columns) + '\n')
        data1 = [epoch_train_loss, epoch_train_acc, epoch_train_f1_score,
                 epoch_valid_loss, epoch_valid_acc, epoch_valid_f1_score]
        data1 = ['{:4f}'.format(d) for d in data1]
        f.write(','.join(data1) + '\n')
        f.close()
    else:
        f = open(out_path, 'at', encoding='utf-8-sig')
        data1 = [epoch_train_loss, epoch_train_acc, epoch_train_f1_score,
                 epoch_valid_loss, epoch_valid_acc, epoch_valid_f1_score]
        data1 = ['{:4f}'.format(d) for d in data1]
        f.write(','.join(data1) + '\n')
        f.close()


def save_block_csv(index, epoch, block_train_loss, block_train_acc, block_train_f1_score,  out_path):
    """
    此文件用户储存训练过程中训练集与每个子集的loss,准确率,f1-score
    """
    if not os.path.exists(out_path) or (index == 0 and epoch == 1):
        f = open(out_path, 'wt', encoding='utf-8-sig')
        columns = ['index', 'train_loss', 'train_acc', 'train_f1_score']
        f.write(','.join(columns) + '\n')
    else:
        f = open(out_path, 'at', encoding='utf-8-sig')
    data1 = [block_train_loss, block_train_acc, block_train_f1_score]
    data2 = [str(index)] + ['{:4f}'.format(d) for d in data1]
    f.write(','.join(data2) + '\n')
    f.close()


def classification_metrics(y_true, y_pred):
    """ 分类任务的评价指标， 传入的数据需要是 numpy 类型的 """

    acc = metrics.accuracy_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred, average='macro')
    return acc, f1_score


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def set_random_seed(args):
    """
    设置训练的随机种子
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
