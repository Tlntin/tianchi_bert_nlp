## 使用说明
- 找个预训练的bert丢data/pre_model_dir文件夹，然后运行main.py就可以了
- 模型下载地址[跳转](https://github.com/ymcui/Chinese-BERT-wwm)
- 代码说明地址：[跳转](https://tianchi.aliyun.com/notebook-ai/detail?postId=160581)

### 文件树

```shell
├── BertRCNN   # 模型文件
│   ├── args.py  # 模型的默认参数，不用管它
│   └── BertRCNN.py  # 模型
├── config.py  # 配置文件，看看你想改啥，最好不要动文件
├── data  # 原始数据
│   ├── NLP_A_Data1128.md
│   ├── pre_model_dir  # 需要你把模型下载丢进去
│   │   ├── bert_config.json
│   │   ├── pytorch_model.tar.gz  # 这是一个gz结尾的文件，如果你下载下来没有那就把bert_config.json与pytorch_model.bin压缩成这个文件
│   │   ├── special_tokens_map.json
│   │   └── vocab.txt
│   ├── submit_sample.zip
│   ├── test
│   │   ├── OCEMOTION_a.csv
│   │   ├── OCNLI_a.csv
│   │   └── TNEWS_a.csv
│   └── train
│       ├── OCEMOTION_train1128.csv
│       ├── OCNLI_train1128.csv
│       └── TNEWS_train1128.csv
├── main.py  # 主运行文件
├── Models  # 其它开源的模型文件，不用管
│   ├── Conv.py
│   ├── Embedding.py
│   ├── Highway.py
│   ├── Linear.py
│   └── LSTM.py
├── out_data  # 输出路径
└── utils  # 重要的代码在这里
    ├── block_loss_weight.py  # 利用block数据调整loss权重
    ├── data_preprocess.py  # 数据预处理，最最基础的操作
    ├── dataset.py  # 构建dataset与dataloader
    ├── epoch_loss_weight.py  # 利用epoch的训练结果调整loss权重
    ├── facal_loss.py  # facal-loss，据说可以解决数据不平衡问题，试了貌似没用
    ├── test.py  # 测试代码并生成zip文件，用于几条结果
    ├── tools.py  # 储存了一些小工具，比如计算参数，保存模型等等
    └── train_valid.py  # 训练、验证代码
```

### 模型得分情况

| 模型类型              | 本地valid-f1-score | 提交(A榜)test-f1-score |
| --------------------- | ------------------ | ---------------------- |
| BERT-base             | 0.594              | 0.628                  |
| RoBERTa-wwm-ext-large | 0.614              | 0.638                  |

- 插播一则新闻：各位大佬有没有招NLP算法的，Base广州，转行一年（也就是没啥经验），中小厂就行，有的话可以加本人v`**Tlntin_**`，谢谢各位大佬了。
- 个人技能图谱[查看](https://mm.edrawsoft.cn/map.html?sharecode=5fe6c20e56c771a95948896),有合适的帮忙推荐一下工作哈。
