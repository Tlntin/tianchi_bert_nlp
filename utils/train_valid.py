import time
import torch
import os
import random
import numpy as np
from tqdm import tqdm
from config import Config
from utils.dataset import get_dataloader
from utils.epoch_loss_weight import EpochLossWeight
from utils.block_loss_weight import BlockLossWeight
from utils.tools import save_epoch_csv, save_checkpoint, classification_metrics, clip_gradient
from utils.tools import save_block_csv

config = Config


def normal_block(dataloader, model, criterion, is_train, data_describe, step_grad,
                 describe, file_name: str, loss_weights_dict: dict, optimizer=None):
    """
    通用部分
    :param dataloader: 数据集
    :param model: 模型
    :param criterion: loss计算函数
    :param is_train: 是否为训练模式
    :param data_describe: 用tqdm封装的迭代器
    :param step_grad 累计多少个batch_size更新一次
    :param describe: 描述性话，用于tqdm输出
    :param file_name: 数据集的文件名，这个主要获取label
    :param loss_weights_dict: 主要用于获取动态权重系数,
    :param optimizer: 优化器
    """
    total_loss = 0
    total_correct = 0  # 记录预测正确的值
    total_num = 0  # 记录总数
    total_pred_list = []  # 记录预测效果
    total_label_list = []  # 记录实际标签结果
    if data_describe is None:
        iter_data = tqdm(enumerate(dataloader), total=len(dataloader))
        data_describe = iter_data
    else:
        iter_data = enumerate(dataloader)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    for step, (input_id, segment_id, input_mask, label) in iter_data:
        temp_size = input_id.shape[0]
        input_id, segment_id, input_mask, label = input_id.to(device), segment_id.to(device), \
                                                  input_mask.to(device), label.to(device)
        total_num += temp_size
        label_num = config.label_num_dict[file_name]
        label_num = torch.from_numpy(np.array([label_num])).to(device)  # 必须加载到device，否则无法训练
        outputs = model(label_num, input_id, segment_id, input_mask)
        loss = criterion(outputs, label)
        # # 如果存着梯度累积,则loss需要平均一下
        # if config.gradient_accumulation_steps > 1:
        #     loss = loss / config.gradient_accumulation_steps
        pred_list = torch.argmax(outputs, 1)
        total_pred_list.extend(pred_list.detach().cpu().data.numpy())  # 统计所有预测值
        total_label_list.extend(label.detach().cpu().data.numpy())
        if is_train:
            loss2 = loss * torch.tensor(loss_weights_dict[file_name]).to(device)
            loss2.backward()  # 变更为loss2，不影响实际loss的值，只影响grad
            # 每隔累计个梯度才更新一次，或者最后一个step也要更新一下参数，不然会出问题
            if (step + 1) % step_grad == 0 or (step + 1) == len(dataloader):
                clip_gradient(optimizer, 1.0)  # 切割梯度
                optimizer.step()
                optimizer.zero_grad()
        # 计算temp_loss与temp_acc
        total_loss += loss.item() * temp_size
        temp_correct = torch.sum(torch.eq(pred_list.cpu().data, label.cpu().data))  # 统计预测正确的值
        total_correct += temp_correct.float()  # 转成浮点型
        data_describe.set_description('{}: loss:{:.4f}, acc:{:.2f}%'.\
                                      format(describe, loss.item(), temp_correct.float() * 100 / temp_size))
    return total_loss, total_correct, total_num, total_pred_list, total_label_list


def train_epoch(model, criterion, optimizer, epoch, epoch_loss_weights_dict):
    """
    训练模型, 单个epoch
    :param model: 模型
    :param criterion: 损失函数
    :param optimizer: 优化器,控制梯度下降
    :param epoch: 第n次训练
    :param epoch_loss_weights_dict: 主要用于获取动态权重系数
    """
    # --- 训练模式 --- #
    model.train()
    epoch_train_label_dict = {}  # 记录一个epoch下的所有train_label
    epoch_train_pred_label_dict = {}
    epoch_train_loss_dict = {}
    epoch_train_num_dict = {}
    # 初始化epoch字典
    for file_name in config.file_name_list:
        epoch_train_label_dict[file_name] = []
        epoch_train_pred_label_dict[file_name] = []
        epoch_train_loss_dict[file_name] = 0
        epoch_train_num_dict[file_name] = 0
    # 构建子集的索引
    train_file_num = len(os.listdir(os.path.join(config.train_split_dir, config.file_name_list[0])))
    index_list1 = list(range(1, train_file_num + 1))
    index_list2 = list(range(1, train_file_num + 1))
    index_list3 = list(range(1, train_file_num + 1))
    # 打散三个数据集的索引,防止每次训练数据都一样
    random.shuffle(index_list1)
    random.shuffle(index_list2)
    random.shuffle(index_list3)
    # 训练子数据集然后将数据汇总
    data_iter = tqdm(enumerate(zip(index_list1, index_list2, index_list3)), total=len(index_list1))
    block_loss_weight = BlockLossWeight()
    for set_idx, index_list in data_iter:
        block_loss_weights_dict = block_loss_weight.run()
        result_loss_weight_dict = {}
        for file_name1 in config.file_name_list:
            value1 = epoch_loss_weights_dict[file_name1]
            value2 = block_loss_weights_dict[file_name1]
            result_loss_weight_dict[file_name1] = 0.5 * value1 + 0.5 * value2
        for file_name, i in zip(config.file_name_list, index_list):
            batch_size = config.batch_size_dict[file_name]
            file_path = os.path.join(config.train_split_dir, file_name, f'{i}.csv')
            sub_dataloader = get_dataloader(file_path, batch_size, True)
            describe = f'训练集 epoch:{epoch}/{config.epochs} 数据集：{file_name}：{set_idx}/{train_file_num}'
            step_grad = config.gradient_accumulation_step_dict[file_name]
            train_loss, train_correct, train_num, train_pred_list, train_label_list =\
                normal_block(sub_dataloader, model, criterion, True, data_iter, step_grad, describe,
                             file_name, result_loss_weight_dict, optimizer)
            epoch_train_label_dict[file_name].extend(train_label_list)
            epoch_train_pred_label_dict[file_name].extend(train_pred_list)
            epoch_train_loss_dict[file_name] += train_loss
            epoch_train_num_dict[file_name] += train_num
            # --- 临时加入block_数据保存与block数据权重计算
            block_train_loss = train_loss / train_num
            block_train_acc, block_train_f1_score = classification_metrics(train_label_list, train_pred_list)
            # 输出路径
            out_dir = os.path.join(config.out_data_dir, 'block_dir')
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            out_path = os.path.join(out_dir, f'out_{file_name}.csv')
            save_block_csv(set_idx, epoch, block_train_loss, block_train_acc, block_train_f1_score, out_path)
        # # 临时断开,测试一下程序效果，查看隐藏bug
        # break
    # 统计每个数据集的loss,acc,f1_score，然后求平均计算总的，每个都保存一下
    train_loss_list = []
    train_acc_list = []
    train_f1_score_list = []
    for file_name in config.file_name_list:
        epoch_train_loss = epoch_train_loss_dict[file_name] / epoch_train_num_dict[file_name]
        epoch_train_acc, epoch_train_f1_score = classification_metrics(epoch_train_label_dict[file_name],
                                                                       epoch_train_pred_label_dict[file_name])
        train_loss_list.append(epoch_train_loss)
        train_acc_list.append(epoch_train_acc)
        train_f1_score_list.append(epoch_train_f1_score)
    return train_loss_list, train_acc_list, train_f1_score_list


def valid_epoch(model, criterion, epoch):
    """
    验证模型,单个epoch
    :param
    """
    # -- 验证模式 -- #
    model.eval()
    valid_loss_list = []
    valid_acc_list = []
    valid_f1_score_list = []
    with torch.no_grad():
        for file_name in config.file_name_list:
            batch_size = config.batch_size_dict[file_name]
            file_path = os.path.join(config.processed_valid, f'{file_name}.csv')
            valid_dataloader = get_dataloader(file_path, batch_size, False)
            step_grad = config.gradient_accumulation_step_dict[file_name]
            describe = f'验证集 epoch:{epoch}/{config.epochs} 数据集：{file_name} '
            valid_loss, valid_correct, valid_num, valid_pred_list, valid_label_list = \
                normal_block(valid_dataloader, model, criterion, False, None,
                             step_grad, describe, file_name, {})
            epoch_valid_loss = valid_loss / valid_num  # 这里必须除，因为是总数，别漏掉了
            epoch_valid_acc, epoch_valid_f1_score = classification_metrics(valid_label_list, valid_pred_list)
            valid_loss_list.append(epoch_valid_loss)
            valid_acc_list.append(epoch_valid_acc)
            valid_f1_score_list.append(epoch_valid_f1_score)
        return valid_loss_list, valid_acc_list, valid_f1_score_list


def train_valid(model, criterion, optimizer):
    """
    总训练与验证程序
    """
    best_score = 0  # 最佳得分，目前用的f1-score
    epochs_since_improvement = 0  # 如果没有改善，累计次数，累计一定次数就退出
    start_epoch = 1  # 开始的epoch
    # 如果存在上次的训练记录
    since = time.time()
    if os.path.exists(config.checkpoint_file):
        print('发现模型', config.checkpoint_file)
        checkpoint = torch.load(config.checkpoint_file)
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint['best_score']
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    # 初始化loss权重
    epoch_loss_weights = EpochLossWeight()
    # 加入block输出
    block_dir = os.path.join(config.out_data_dir, 'block_dir')
    if not os.path.exists(block_dir):
        os.mkdir(block_dir)
    else:
        for file in os.listdir(block_dir):
            os.remove(os.path.join(block_dir, file))
    # --- 正式训练
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_loss_weights_dict = epoch_loss_weights.run()
        train_loss_list, train_acc_list, train_f1_score_list = train_epoch(model, criterion, optimizer,
                                                                           epoch, epoch_loss_weights_dict)
        valid_loss_list, valid_acc_list, valid_f1_score_list = valid_epoch(model, criterion, epoch)
        # 计算平均表现情况
        train_loss_average = np.average(train_loss_list)
        train_acc_average = np.average(train_acc_list)
        train_f1_score_average = np.average(train_f1_score_list)
        valid_loss_average = np.average(valid_loss_list)
        valid_acc_average = np.average(valid_acc_list)
        valid_f1_score_average = np.average(valid_f1_score_list)
        # 保存记录
        out_path = os.path.join(config.out_data_dir, 'out_average.csv')  # 输出路径
        save_epoch_csv(epoch, train_loss_average, train_acc_average, train_f1_score_average,
                       valid_loss_average, valid_acc_average, valid_f1_score_average, out_path)
        # 保存三个数据集所有数据到csv
        for file_name, train_loss, train_acc, train_f1, valid_loss, valid_acc, valid_f1 \
                in zip(config.file_name_list, train_loss_list, train_acc_list, train_f1_score_list,
                       valid_loss_list, valid_acc_list, valid_f1_score_list):
            out_path = os.path.join(config.out_data_dir, f'out_{file_name}.csv')  # 输出路径
            save_epoch_csv(epoch, train_loss, train_acc, train_f1, valid_loss, valid_acc, valid_f1, out_path)
        print('epoch: {} / {} train_loss:{:.4f}, train_acc:{:.2f}%  train_f1_score:{:.4f}\n\
        valid_loss:{:.4f}, valid_acc:{:.2f}% valid_f1_score:{:.4f}'.\
              format(epoch, config.epochs, train_loss_average, train_acc_average * 100, train_f1_score_average,
                     valid_loss_average, valid_acc_average * 100, valid_f1_score_average))
        # -- 保存最佳模型，以valid f1为准 -- #
        is_best = valid_f1_score_average > best_score
        best_score = max(valid_f1_score_average, best_score)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            if epochs_since_improvement >= config.break_epoch:
                break
        else:
            best_score = valid_f1_score_average
            epochs_since_improvement = 0
        time_elapsed = time.time() - since  # 计算时间间隔
        print('当前训练共用时{:.0f}时{:.0f}分{:.0f}秒'.format(time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60))
        loss_weights_dict = epoch_loss_weights.run()
        print('当前loss权重为：', loss_weights_dict)
        print('目前最高的F1-score为：{:.6f}'.format(best_score))
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_score, is_best)
    return model, optimizer


def train_enhance(model, criterion, optimizer, loss_weights_dict):
    """
    用于训练集的增强训练，一般用于最后一次训练
    用验证集去训练模型
    :param model: 模型
    :param criterion: 损失函数
    :param optimizer: 优化器,控制梯度下降
    :param loss_weights_dict: 主要用于获取动态权重系数
    """
    # 训练模式
    model.train()
    valid_file_num = len(os.listdir(os.path.join(config.valid_split_dir, config.file_name_list[0])))
    # 构建子集的索引
    data_iter = tqdm(range(1, valid_file_num + 1))
    for idx in data_iter:
        for file_name in config.file_name_list:
            file_path = os.path.join(config.valid_split_dir, file_name, f'{idx}.csv')
            batch_size = config.batch_size_dict2[file_name]  # 专用batch_size
            step_grad = config.gradient_accumulation_step_dict2[file_name]  # 专用step_grad
            sub_dataloader = get_dataloader(file_path, batch_size, False)
            describe = f'增强训练中 数据集：{file_name}：{idx}/{valid_file_num}'
            normal_block(sub_dataloader, model, criterion, True, data_iter, step_grad,
                         describe, file_name, loss_weights_dict, optimizer)
    return model









