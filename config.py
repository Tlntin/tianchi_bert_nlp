import os


class Config:
    learning_rate = 2e-5
    warmup_proportion = 0.1
    embedding_dim = 300
    hidden_size = 300
    epochs = 20  # 训练20次
    break_epoch = 5  # 连续5次没有改善就跳出训练
    file_name_list = ['OCNLI', 'OCEMOTION', 'TNEWS']  # 文件名
    label_num_dict = {'OCNLI': 3, 'OCEMOTION': 7, 'TNEWS': 15}
    max_seq_num_dict = {'OCNLI': 120, 'OCEMOTION': 312, 'TNEWS': 150}
    batch_size_dict = {'OCNLI': 60, 'OCEMOTION': 20, 'TNEWS': 38}
    gradient_accumulation_step_dict = {'OCNLI': 5, 'OCEMOTION': 16, 'TNEWS': 8}  # 每隔n个更新一次梯度
    batch_size_dict2 = {'OCNLI': 40, 'OCEMOTION': 15, 'TNEWS': 25}
    gradient_accumulation_step_dict2 = {'OCNLI': 5, 'OCEMOTION': 16, 'TNEWS': 8}  # 每隔n个更新一次梯度
    seed = 10  # 设置随机种子
    filter_num = 100  # 也就是out_channel
    filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 过滤器
    #  --- 原始存放数据 -- #
    raw_data_dir = os.path.join(os.path.dirname(__file__), 'data')  # 原始存放数据文件夹
    train_dir = os.path.join(raw_data_dir, 'train')
    test_dir = os.path.join(raw_data_dir, 'test')
    pre_model_dir = os.path.join(raw_data_dir, 'pre_model_dir')  # 预训练模型路径
    # --- 输出数据存放路径 --- #
    out_data_dir = os.path.join(os.path.dirname(__file__), 'out_data')  # 输出数据文件夹
    out_model_dir = os.path.join(out_data_dir, 'model')  # 模型保存路径
    # 处理好的数据存放文件夹
    processed_data_dir = os.path.join(out_data_dir, 'processed_data')
    processed_total = os.path.join(processed_data_dir, 'total')
    processed_train = os.path.join(processed_data_dir, 'train')
    processed_valid = os.path.join(processed_data_dir, 'valid')
    processed_test = os.path.join(processed_data_dir, 'test')
    for dir1 in [raw_data_dir, train_dir, test_dir, pre_model_dir, out_data_dir, processed_data_dir,
                 out_model_dir, processed_total, processed_train, processed_valid, processed_test]:
        if not os.path.exists(dir1):
            os.mkdir(dir1)
    # -- BERT预训练模型相关参数 -- #
    pre_model_json_path = os.path.join(pre_model_dir, 'bert_config.json')
    pre_model_path = os.path.join(pre_model_dir, 'bert-base-chinese.tar.gz')
    pre_model_vocab_path = os.path.join(pre_model_dir, 'bert-base-chinese-vocab.txt')
    checkpoint_file = os.path.join(out_model_dir, 'checkpoint.pth')  # 每次训练一次模型都保存一次，可以重复训练
    checkpoint_config = os.path.join(out_model_dir, 'model_config.json')
    best_checkpoint_file = os.path.join(out_model_dir, 'best_checkpoint.pth')  # 最佳valid_loss保存路径
    best_checkpoint_config = os.path.join(out_model_dir, 'best_model_config.json')
    label2id_path = os.path.join(out_data_dir, 'label2id.json')
    id2label_path = os.path.join(out_data_dir, 'id2label.json')

    # -- 分割数train/valid数据集，主要是为了三个数据集合并计算
    train_split_dir = os.path.join(out_data_dir, 'train_split_data')  # 分割后的训练集
    valid_split_dir = os.path.join(out_data_dir, 'valid_split_data')  # 分割后的验证集
    train_split_num = 45  # 暂定将训练集分割成45份
    if not os.path.exists(train_split_dir):
        os.mkdir(train_split_dir)
    if not os.path.exists(valid_split_dir):
        os.mkdir(valid_split_dir)
    for file_name in file_name_list:
        file_dir = os.path.join(train_split_dir, file_name)
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        file_dir2 = os.path.join(valid_split_dir, file_name)
        if not os.path.exists(file_dir2):
            os.mkdir(file_dir2)
