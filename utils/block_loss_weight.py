import pandas as pd
import numpy as np
import os
from config import Config

config = Config()


class BlockLossWeight(object):
    """
    用于动态调整loss的权重,每隔一个Block调整一次
    """
    def __init__(self):
        label_weights_dict = self.get_label_weight()
        d_weights_dict = self.get_d_weight()
        self.init_loss_weight_dict = {}  # 计算初始权重
        for (file_name, d), rate in zip(d_weights_dict.items(), label_weights_dict.values()):
            self.init_loss_weight_dict[file_name] = round(d * rate, 2)
        # print('初始训练权重', self.init_loss_weight_dict)

    @staticmethod
    def get_label_weight():
        """
        计算单个label的权重
        :param
        """
        # 计算每个单个数据集长度
        train_length_dict = {}
        for file_name in config.file_name_list:
            train_df = pd.read_csv(os.path.join(config.processed_train, file_name + '.csv'), index_col=0)
            train_length_dict[file_name] = len(train_df)
        # 计算平均每个label的长度
        one_label_length_dict = {}
        for (file_name, train_length), label_num in zip(train_length_dict.items(), config.label_num_dict.values()):
            length = round(train_length / label_num, 2)
            one_label_length_dict[file_name] = np.log(length)
        # 计算单个label的相对占比系数
        label_length_sum = np.sum(list(one_label_length_dict.values()))
        one_label_rate_dict = {}
        for file_name in config.file_name_list:
            one_label_rate_dict[file_name] = round(one_label_length_dict[file_name] / label_length_sum, 2)
        # 计算单个label的最终得分
        label_weights_dict = {}
        temp_min = np.min(list(one_label_rate_dict.values()))
        for file_name, rate in one_label_rate_dict.items():
            rate1 = round(1 / (rate / temp_min), 2)
            label_weights_dict[file_name] = rate1
        return label_weights_dict

    @staticmethod
    def get_d_weight():
        """
        计算各个数据集的熵值以及权重
        """
        # 计算每个训练集熵值
        d_dict = {}
        for file_name in config.file_name_list:
            train_df = pd.read_csv(os.path.join(config.processed_train, file_name + '.csv'), index_col=0)
            p_list = train_df['label'].value_counts().values / len(train_df)  # 计算每个类别占比
            d = 0
            for p in p_list:
                d = d - p * np.log(p)
            d_dict[file_name] = round(d, 2)
        # 计算熵值的权重
        d_max = np.max(list(d_dict.values()))
        d_weights_dict = {}
        for file_name, d in d_dict.items():
            d2 = round(d / d_max, 2)
            d_weights_dict[file_name] = d2
        return d_weights_dict

    @staticmethod
    def get_f1_score_data():
        """
        计算f1_score最新数据与最近两条数据
        """
        last_f1_score_dict = {}  # 最新f1_score
        all_f1_score_dict = {}
        for file_name in config.file_name_list:
            file_path = os.path.join(config.out_data_dir, 'block_dir', f'out_{file_name}.csv')
            out_df = pd.read_csv(file_path)
            length = len(out_df)
            if length == 1:
                f1_score = out_df.loc[length - 1, 'train_f1_score']
                f1_score_list = [f1_score]
            else:
                f1_score_list = out_df.loc[length - 2:, 'train_f1_score'].values.tolist()
                all_f1_score_dict[file_name] = f1_score_list
            last_f1_score_dict[file_name] = f1_score_list[-1]  # 统计最新值
        return last_f1_score_dict, all_f1_score_dict

    @staticmethod
    def get_now_score_weight(last_f1_score_dict):
        """
        获取最新f1_score的权重
        """
        f1_score_min = np.min(list(last_f1_score_dict.values()))
        last_score_dict = {}
        for file_name, f1_score in last_f1_score_dict.items():
            score = round(1 / (f1_score / f1_score_min), 2)
            last_score_dict[file_name] = score
        return last_score_dict

    @staticmethod
    def get_update_score_weight(all_f1_score_dict):
        """
        获取最近更新的f1-score权重变化
        :param
        """
        # 获取f1-score的增长
        f1_score_rise_dict = {}
        for file_name, f1_score_list in all_f1_score_dict.items():
            f1_score_rise_dict[file_name] = round(f1_score_list[-1] / f1_score_list[-2] - 1, 4)
        f1_score_rise_mean = np.mean(list(f1_score_rise_dict.values()))
        f1_score_rise_max = np.max(list(f1_score_rise_dict.values()))
        # 计算增长的权重系数
        f1_score_rise_dict2 = {}
        for file_name, score_rise in f1_score_rise_dict.items():
            score = round((score_rise + f1_score_rise_mean) / (f1_score_rise_max + f1_score_rise_mean), 2)
            f1_score_rise_dict2[file_name] = score
        return f1_score_rise_dict2

    @staticmethod
    def clip_weight(weight_dict, min_weight=0.1, max_weight=1.0):
        """
        切割权重
        :param weight_dict: 权重字典
        :param min_weight: 最小权重,默认为0.1
        :param max_weight: 最大权重，默认为1
        """
        weight_dict2 = {}
        for file_name, weight in weight_dict.items():
            weight = max(min_weight, weight)
            weight = min(max_weight, weight)
            weight_dict2[file_name] = weight
        return weight_dict2

    def run(self):
        """
        正式运行
        :param
        """
        loss_weight_dict = {}
        out_dir = os.path.join(config.out_data_dir, 'block_dir')
        if len(os.listdir(out_dir)) < 3:
            loss_weight_dict = self.init_loss_weight_dict
        else:
            last_f1_score_dict, all_f1_score_dict = self.get_f1_score_data()
            now_score_weights = self.get_now_score_weight(last_f1_score_dict)
            if len(all_f1_score_dict) == 0:
                for (file_name, init_score), now_score in zip(self.init_loss_weight_dict.items(),
                                                              now_score_weights.values()):
                    loss_weight_dict[file_name] = round(init_score * 0.5 + now_score * 0.5, 2)
            else:
                rise_score_dict = self.get_update_score_weight(all_f1_score_dict)
                for (file_name, now_score), rise_score in zip(now_score_weights.items(), rise_score_dict.values()):
                    init_score = self.init_loss_weight_dict[file_name]
                    loss_weight_dict[file_name] = round(init_score * 0.2 + now_score * 0.5 + rise_score * 0.3, 2)
        return self.clip_weight(loss_weight_dict)


if __name__ == '__main__':
    loss_weight = BlockLossWeight()
    dict1 = loss_weight.run()
    print(dict1)
    dict1 = {'OCNLI': -0.08, 'OCEMOTION': 1.0, 'TNEWS': 0.66}
    dict2 = loss_weight.clip_weight(dict1)
    print(dict2)

