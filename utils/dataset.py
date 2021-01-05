from config import Config
from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

config = Config()


class MyDataset(Dataset):
    """
    构建自定义数据集，继承于pytorch
    """

    def __init__(self, file_path):
        """
        初始化
        :file_path:文件路径
        """
        super().__init__()
        df1 = pd.read_csv(file_path, index_col=0)
        self.input_ids = df1['corpus'].apply(self.parse_text).values
        self.segment_ids = df1['segment_ids'].apply(self.parse_text).values
        self.input_masks = df1['input_mask'].apply(self.parse_text).values
        if 'label' in df1.columns:
            self.label_ids = df1['label'].values
        else:
            self.label_ids = None

    def __getitem__(self, index):
        """
        该方法用于获取一个子元素，index由数据集自动迭代得到
        """
        input_id = torch.from_numpy(self.input_ids[index])
        segment_id = torch.from_numpy(self.segment_ids[index])
        input_mask = torch.from_numpy(self.input_masks[index])
        if self.label_ids is not None:
            label = self.label_ids[index]
            return input_id, segment_id, input_mask, label
        else:
            return input_id, segment_id, input_mask

    def __len__(self):
        return len(self.input_ids)

    @staticmethod
    def parse_text(text):
        """
        用于处理原始句子，将其变成tensor
        :param text: 单个句子
        """
        corpus = text.split(" ")  # 句子切割成词列表,已提前转了id，所以这里不用再转id
        corpus = [int(w) for w in corpus]  # 转str为数字
        corpus = np.array(corpus, dtype=np.int64)
        # corpus = torch.from_numpy(np.array(corpus, dtype='int64'))
        return corpus


def get_dataloader(file_path: str, batch_size, is_shuffle: bool = False):
    """
    获取数据集
    :param file_path: 文件路径
    :param batch_size: 批量大小
    :param is_shuffle: 是否打散
    """
    dataset = MyDataset(file_path)
    dataloader = DataLoader(dataset, shuffle=is_shuffle, batch_size=batch_size)
    return dataloader


if __name__ == '__main__':
    file_path1 = os.path.join(config.processed_train, config.file_name_list[0] + '.csv')
    batch_size1 = config.batch_size_dict[config.file_name_list[0]]
    train_dataloader = get_dataloader(file_path1, batch_size1, True)
    for input_id1, segment_id1, input_mask1, label1 in train_dataloader:
        print(input_id1.shape, segment_id1.shape, input_mask1.shape, label1.shape)
        break
    for input_id1, segment_id1, input_mask1, label1 in tqdm(train_dataloader):
        pass

