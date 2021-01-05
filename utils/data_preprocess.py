import pandas as pd
from config import Config
from tqdm import tqdm
import os
import json
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

config = Config()


class DataPreProcess(object):
    """
    数据预处理,主要包括分词，转字典id操作
    """
    def __init__(self):
        """
        """
        # 初始化tokenizer
        self.tokenizer = BertTokenizer(vocab_file=config.pre_model_vocab_path)
        print('tokenizer初始化完成')

    @staticmethod
    def truncate_seq_pair(tokens_a, tokens_b, max_length):
        """
        用于同时削减2个句子的长度
        """
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        return tokens_a, tokens_b

    def token2id(self, text_a, text_b, max_seq_length):
        """
        将所有原始文本转成语料库id，目前为字模型，后期考虑转成为词模型
        :param text_a: 第一个文本语料,单个句子,或者为多个词构建的列表也行，默认为字模型，也就是单个句子
        :param text_b: 第二个文本语料可有可无，单个句子，,或者为多个词构建的列表也行
        :param max_seq_length: 最大句子长度
        """
        # --- 第一步，对text_a与text_b分别构建语料 --- #
        token_a = []
        token_b = []
        for word1 in text_a:
            token_a.append(self.tokenizer.convert_tokens_to_ids(word1))
        # 如果不存在第二个句子的话
        if not bool(text_b):
            # 根据最大句子长度，只削减token_a
            if len(token_a) > max_seq_length - 2:  # 减2是为了考虑cls与sep
                token_a = token_a[: max_seq_length - 2]
                # print('发生字符切割')
                # raise Exception('发生单个句子的字符切割')
        else:
            for word2 in text_b:
                token_b.append(self.tokenizer.convert_tokens_to_ids(word2))
            # 根据最大句子分别削减token_a与token_b
            if len(token_a) + len(token_b) > max_seq_length - 3:  # 减3是为了考虑cls与2个sep
                token_a, token_b = self.truncate_seq_pair(token_a, token_b, max_seq_length - 3)
                # print('发生字符切割')
                # raise Exception('发生两个句子的字符切割')
        return token_a, token_b

    def parse_token_id(self, token_a, token_b, max_seq_length):
        cls_id = self.tokenizer.convert_tokens_to_ids('[CLS]')   # 开始符号
        seq_id = self.tokenizer.convert_tokens_to_ids('[SEP]')  # 句子结束符号
        corpus = [cls_id]  # 记录语料id
        segment_ids = []  # 记录句子编号id
        corpus.extend(token_a)
        corpus.append(seq_id)  # 代表句子结束
        segment_ids.extend([0] * (len(token_a) + 2))  # 第一个句子,+2代表cls与seq
        if len(token_b) > 0:
            corpus.extend(token_b)
            corpus.append(seq_id)
            segment_ids.extend([1] * (len(token_b) + 1))  # 第二个句子 + 1代表seq
        input_mask = [1] * len(corpus)  # 隐码id
        # print(len(corpus), max_seq_length)
        if len(corpus) < max_seq_length:
            input_mask.extend([0] * (max_seq_length - len(corpus)))  # 0为隐码
            segment_ids.extend([0] * (max_seq_length - len(corpus)))  # 默认补零
            corpus.extend([0] * (max_seq_length - len(corpus)))  # padding最后补全
        assert len(corpus) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        return corpus, segment_ids, input_mask
    
    def get_corpus(self, temp_df: pd.DataFrame, text_num: int, max_seq_length):
        data_list = []
        idx_list = tqdm(range(len(temp_df)))
        for i in idx_list:
            idx_list.set_description('正在构建语料库')
            if text_num == 1:
                text1 = temp_df.loc[i, 'text']
                text2 = None
            else:
                text1 = temp_df.loc[i, 'text1']
                text2 = temp_df.loc[i, 'text2']
            token_a, token_b = self.token2id(text1, text2, max_seq_length)
            corpus, segment_ids, input_mask = self.parse_token_id(token_a, token_b, max_seq_length)
            corpus = [str(c) for c in corpus]
            segment_ids = [str(c) for c in segment_ids]
            input_mask = [str(c) for c in input_mask]
            corpus = ' '.join(corpus)
            segment_ids = ' '.join(segment_ids)
            input_mask = ' '.join(input_mask)
            if 'label' in temp_df.columns:
                label = temp_df.loc[i, 'label']
                data_list.append([corpus, segment_ids, input_mask, label])
            else:
                data_list.append([corpus, segment_ids, input_mask])
        if len(data_list[0]) == 4:
            df1 = pd.DataFrame(data_list, columns=['corpus', 'segment_ids', 'input_mask', 'label'])
        else:
            df1 = pd.DataFrame(data_list, columns=['corpus', 'segment_ids', 'input_mask'])
        return df1

    @staticmethod
    def label2id(temp_df: pd.DataFrame):
        """
        将标签直转化为数字id
        """
        label_list = temp_df['label'].unique().tolist()
        label_list.sort()
        label2id = {label: idx for idx, label in enumerate(label_list)}
        id2label = {idx: label for idx, label in enumerate(label_list)}
        temp_df['label'] = temp_df['label'].map(label2id)
        return temp_df, label2id, id2label

    def process_file(self, file_path: str, is_train: bool, text_num: int, max_seq_length):
        """
        读取csv操作，并且将csv转成token_id,存储到csv中
        :param file_path: 文件所在路径
        :param is_train: 是否为训练集
        :param text_num: 每个数据拥有的句子，1或者2
        :param max_seq_length: 最大句子长度，默认为最大的句子长度，后期可以尝试不同句子长度
        """
        if text_num == 1:
            text_column = ['text']
        elif text_num == 2:
            text_column = ['text1', 'text2']
        else:
            raise Exception('text num must be 1 or 2')
        if is_train:
            columns = text_column + ['label']
        else:
            columns = text_column
        temp_df = pd.read_csv(file_path, names=columns, index_col=0, sep='\t', quoting=3)
        label2id = None
        id2label = None
        # 加入label转换
        if is_train:
            temp_df, label2id, id2label = self.label2id(temp_df)
        result_df = self.get_corpus(temp_df, text_num, max_seq_length)
        return result_df, label2id, id2label

    @staticmethod
    def split_total_data(total_df: pd.DataFrame, test_size=0.2, data_describe=None, description=None):
        """
        对total_df进行分割，实际上就是把原有的train_df分割成train_df与valid_df
        :param total_df: 总数据集
        :param test_size: 分割大小，如果是小数那就是比例，如果是int那就是按这个数分割
        :param data_describe: 描述，方便打印进度
        :param description: 描述的内容
        """
        train_idx = total_df.index.values
        train_labels = total_df.label.values
        train_id, valid_id, train_label, valid_label = train_test_split(train_idx, train_labels,
                                                                        test_size=test_size, random_state=2)
        str1 = '训练集长度为：{}，验证集长度为：{}'.format(len(train_id), len(valid_id))
        if data_describe is None:
            print('正在划分总数据集', str1)
        else:
            data_describe.set_description(description + str1)

        # -- 重构训练集 -- #
        if data_describe is None:
            print('正在重构训练集')
        train_data = []
        for idx in train_id:
            data = total_df.loc[idx].values
            train_data.append(data)
        train_df = pd.DataFrame(train_data, columns=total_df.columns)
        if data_describe is None:
            print('正在构建验证集')
        # -- 构建验证集 -- #
        valid_data = []
        for idx in valid_id:
            data2 = total_df.loc[idx].values
            valid_data.append(data2)
        valid_df = pd.DataFrame(valid_data, columns=train_df.columns)
        return train_df, valid_df

    @staticmethod
    def clear_split_data_dir():
        """
        清空分割后的训练数据
        """
        for file_name in config.file_name_list:
            file_dir = os.path.join(config.train_split_dir, file_name)
            if not os.path.exists(file_dir):
                os.mkdir(file_dir)
            else:
                file_list = os.listdir(file_dir)
                for file1 in file_list:
                    os.remove(os.path.join(file_dir, file1))

    @staticmethod
    def get_sub_num(train_df: pd.DataFrame, file_name, split_num=config.train_split_num):
        """
        分割训练集，计算每个子集大小
        :param train_df: 训练集
        :param file_name: 数据集文件名，取自['OCNLI', 'OCEMOTION', 'TNEWS']
        :param split_num: 分割
        :return:sub_num:子集大小
        """
        print('开始划分子训练集')
        batch_size = config.batch_size_dict[file_name]
        if len(train_df) % batch_size != 0:
            batch_num = len(train_df) // batch_size + 1
        else:
            batch_num = len(train_df) // batch_size
        # 计算单个子训练集的长度
        split_rate = round(1 / split_num, 4)
        sub_sample_num = int(batch_num * split_rate)  # 子样本大小
        assert sub_sample_num >= 1
        sub_num = sub_sample_num * batch_size  # 子集大小
        print(f'数据集大小：{len(train_df)}，分割率：{split_rate}，子集大小：{sub_num}')
        return sub_num

    def split_many_data(self, temp_df, sub_num, split_num, output_dir, data_type):
        """
        分割多个数据集
        :param temp_df: 带分割的数据集
        :param sub_num: 子集大小
        :param split_num: 切割后的文件数量
        :param output_dir: 输出路径
        :param data_type: 数据集类型，训练集or验证集
        """
        data_describe = tqdm(range(split_num - 1))
        big_df = temp_df.copy()
        for i in data_describe:
            big_df, small_df = self.split_total_data(big_df, sub_num, data_describe,
                                                     f'{data_type}分割进度:{i+1}/{split_num}\t')
            if i < split_num - 2:
                small_df.to_csv(os.path.join(output_dir, f'{i + 1}.csv'))
            else:
                small_df.to_csv(os.path.join(output_dir, f'{i + 1}.csv'))
                big_df.to_csv(os.path.join(output_dir, f'{i + 2}.csv'))

    def run(self):
        total_label2id = {}  # 记录每个数据集的label-->id方法
        total_id2label = {}  # 记录每个数据集的id-->label方法
        train_split_num_list = []  # 记录训练集切割的子集个数
        valid_split_num_list = []  # 记录切割的子集个数
        sub_num_dict = {}  # 记录训练集/验证集切割的子集长度
        train_df_list = []
        valid_df_list = []
        self.clear_split_data_dir()
        for file_name in config.file_name_list:
            train_file_path = os.path.join(config.train_dir, f"{file_name}_train1128.csv")
            test_file_path = os.path.join(config.test_dir, f'{file_name}_a.csv')
            if file_name == 'OCNLI':
                text_num = 2
            else:
                text_num = 1
            max_seq_num = config.max_seq_num_dict[file_name]
            total_df, label2id, id2label = self.process_file(train_file_path, True, text_num, max_seq_num)
            total_label2id[file_name] = label2id
            total_id2label[file_name] = id2label
            train_df, valid_df = self.split_total_data(total_df)
            train_df_list.append(train_df)
            valid_df_list.append(valid_df)
            # 分割训练集并且保存到对应位置
            sub_num = self.get_sub_num(train_df, file_name)
            sub_num_dict[file_name] = sub_num
            train_split_num = len(train_df) // sub_num if len(train_df) % sub_num == 0 else len(train_df) // sub_num + 1
            train_split_num_list.append(train_split_num)
            valid_split_num = len(valid_df) // sub_num if len(valid_df) % sub_num == 0 else len(valid_df) // sub_num + 1
            valid_split_num_list.append(valid_split_num)
            test_df, _, _ = self.process_file(test_file_path, False, text_num, max_seq_num)
            # 保存文件到对应文件夹
            total_df.to_csv(os.path.join(config.processed_total, f'{file_name}.csv'), index=True, encoding='utf-8')
            train_df.to_csv(os.path.join(config.processed_train, f'{file_name}.csv'), index=True, encoding='utf-8')
            valid_df.to_csv(os.path.join(config.processed_valid, f'{file_name}.csv'), index=True, encoding='utf-8')
            test_df.to_csv(os.path.join(config.processed_test, f'{file_name}.csv'), index=True, encoding='utf-8')
        print('文本转语料完成')
        with open(config.id2label_path, 'wt', encoding='utf-8') as f1:
            json.dump(total_id2label, f1)
        with open(config.label2id_path, 'wt', encoding='utf-8') as f2:
            json.dump(total_label2id, f2)
        print('开始切割训练集，验证集')
        # 取最小切割数量
        min_train_split_num = min(train_split_num_list)
        min_valid_split_num = min(valid_split_num_list)
        for file_name, train_df, valid_df in zip(config.file_name_list, train_df_list, valid_df_list):
            sub_num2 = sub_num_dict[file_name]
            self.split_many_data(train_df, sub_num2, min_train_split_num,
                                 os.path.join(config.train_split_dir, file_name), '训练集')
            self.split_many_data(valid_df, sub_num2, min_valid_split_num,
                                 os.path.join(config.valid_split_dir, file_name), '验证集')
        print('数据集分割完成')


if __name__ == '__main__':
    data_pre = DataPreProcess()
    data_pre.run()



