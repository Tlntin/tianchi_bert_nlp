from config import Config
import torch
import os
from utils.dataset import get_dataloader
from utils.train_valid import train_enhance
from BertRCNN.BertRCNN import BertRCNN
from tqdm import tqdm
import numpy as np
import json
from torch import nn


config = Config()


def test_model(model, out_dir, out_file):
    # --测试模型 -- #
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # 提交结果的预测文件夹
    predict_dir = os.path.join(config.out_data_dir, out_dir)
    if not os.path.exists(predict_dir):
        os.mkdir(predict_dir)
    else:
        file_list = os.listdir(predict_dir)
        file_path_list = [os.path.join(predict_dir, file) for file in file_list]
        for file_path in file_path_list:
            os.remove(file_path)
    with torch.no_grad():
        for file_name in config.file_name_list:
            batch_size = config.batch_size_dict[file_name]
            file_path = os.path.join(config.processed_test, f'{file_name}.csv')
            test_dataloader = get_dataloader(file_path, batch_size, False)
            data_iter = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
            describe = f'开始test模型。数据集：{file_name}\t'
            total_pred_list = []  # 统计所有预测值
            for idx, (input_id, segment_id, input_mask) in data_iter:
                input_id, segment_id, input_mask = input_id.to(device), segment_id.to(device), input_mask.to(device)
                label_num = config.label_num_dict[file_name]
                label_num = torch.from_numpy(np.array([label_num])).to(device)  # 必须加载到device，否则无法训练
                outputs = model(label_num, input_id, segment_id, input_mask)
                pred_list = torch.argmax(outputs, 1)
                total_pred_list.extend(pred_list.detach().cpu().data.numpy())  # 统计所有预测值
                data_iter.set_description(describe + f'{idx}/{len(test_dataloader)}')
            # 将预测值转化为label
            with open(os.path.join(config.out_data_dir, 'id2label.json'), 'rt', encoding='utf-8') as f:
                id2label = json.load(f)
            pred_label_list = [id2label[file_name][str(pred)] for pred in total_pred_list]
            # 将预测值写入json
            file_path = os.path.join(predict_dir, '{}_predict.json'.format(file_name.lower()))
            with open(file_path, 'wt', encoding='utf-8') as f1:
                for idx, pred_label in enumerate(pred_label_list):
                    label_dict = {"id": str(idx), "label": str(pred_label)}
                    f1.write(json.dumps(label_dict, ensure_ascii=False))
                    if idx < len(pred_label_list) - 1:
                        f1.write('\n')
        # 输出压缩文件夹
        out_zip_file = os.path.join(config.out_data_dir, out_file)
        if os.path.exists(out_zip_file):
            os.remove(out_zip_file)
        zip_str = 'cd {}&&zip -q {} ./*'.format(predict_dir, out_zip_file)
        print('压缩ing, ', zip_str)
        os.system(zip_str)
        print('压缩finished')


if __name__ == '__main__':
    device1 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = BertRCNN.from_pretrained(config.pre_model_path, rnn_hidden_size=config.hidden_size).to(device1)
    # 根据最佳模型预测
    checkpoint = torch.load(config.best_checkpoint_file)
    model1.load_state_dict(checkpoint['model_state'])
    test_model(model1, 'best_predict', 'best_predict_submit.zip')