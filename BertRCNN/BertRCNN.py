import torch
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from config import Config
from Models.Linear import Linear
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from pytorch_pretrained_bert.modeling import BertConfig

config = Config()


class BertRCNN(BertPreTrainedModel):

    def __init__(self, args, rnn_hidden_size, num_layers=2, bidirectional=True, dropout=0.2):
        super(BertRCNN, self).__init__(args)
        self.bert = BertModel(args)
        self.apply(self.init_bert_weights)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.rnn_dict = {}
        self.w2_dict = {}
        self.classifier_dict = {}
        # 构建3个分类器，对应三种不同的样本
        label_num_list = list(config.label_num_dict.values())
        # print(label_num_list)
        # 分类器1
        self.rnn1 = nn.LSTM(args.hidden_size, rnn_hidden_size, num_layers,
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.w1 = Linear(args.hidden_size + 2 * rnn_hidden_size, args.hidden_size)
        self.classifier1 = nn.Linear(args.hidden_size, label_num_list[0])
        # 分类器2
        self.rnn2 = nn.LSTM(args.hidden_size, rnn_hidden_size, num_layers,
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.w2 = Linear(args.hidden_size + 2 * rnn_hidden_size, args.hidden_size)
        self.classifier2 = nn.Linear(args.hidden_size, label_num_list[1])
        # 分类器3
        self.rnn3 = nn.LSTM(args.hidden_size, rnn_hidden_size, num_layers,
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.w3 = Linear(args.hidden_size + 2 * rnn_hidden_size, args.hidden_size)
        self.classifier3 = nn.Linear(args.hidden_size, label_num_list[2])

    def forward(self, label_num, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        前向计算
        """
        encoded_layers, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        encoded_layers = self.dropout(encoded_layers)
        # encoded_layers: [batch_size, seq_len, bert_dim]
        label_num = label_num.detach().cpu().data.tolist()[0]  # 获取真正的label
        if label_num == 3:
            outputs, _ = self.rnn1(encoded_layers)
            # outputs: [batch_size, seq_len, rnn_hidden_size * 2]

            x = torch.cat((outputs, encoded_layers), 2)
            # x: [batch_size, seq_len, rnn_hidden_size * 2 + bert_dim]

            y2 = torch.tanh(self.w1(x)).permute(0, 2, 1)
            # y2: [batch_size, rnn_hidden_size * 2, seq_len]

            y3 = F.max_pool1d(y2, y2.size()[2]).squeeze(2)

            logits = self.classifier1(y3)
        elif label_num == 7:
            outputs, _ = self.rnn2(encoded_layers)
            x = torch.cat((outputs, encoded_layers), 2)
            y2 = torch.tanh(self.w2(x)).permute(0, 2, 1)
            y3 = F.max_pool1d(y2, y2.size()[2]).squeeze(2)
            logits = self.classifier2(y3)
        else:
            outputs, _ = self.rnn3(encoded_layers)
            x = torch.cat((outputs, encoded_layers), 2)
            y2 = torch.tanh(self.w3(x)).permute(0, 2, 1)
            y3 = F.max_pool1d(y2, y2.size()[2]).squeeze(2)
            logits = self.classifier3(y3)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertRCNN.from_pretrained(config.pre_model_path, rnn_hidden_size=config.hidden_size).to(device)
    for file_name1 in config.file_name_list:
        sequence_num = config.max_seq_num_dict[file_name1]
        input_id = torch.randint(0, 100, size=(1, sequence_num)).to(device)
        segment_id = torch.randint(0, 1, size=(1, sequence_num)).to(device)
        input_mask = torch.randint(0, 1, size=(1, sequence_num)).to(device)
        label_num1 = config.label_num_dict[file_name1]
        label_num2 = torch.from_numpy(np.array([label_num1])).to(device)
        # print(model)
        print(f'{file_name1}输入维度为：', input_id.shape)
        outputs1 = model(label_num2, input_id, segment_id, input_mask)
        print(f'{file_name1}输出维度：', outputs1.shape)
