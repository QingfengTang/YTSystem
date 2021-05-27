from torch import nn
from fastNLP.modules import LSTM
import torch
import torch.nn.functional as F

class BiLSTMMaxPoolCls(nn.Module):
    def __init__(self, embed, num_classes, hidden_size=200, num_layers=1, dropout=0.5):
        super().__init__()
        self.embed = embed

        self.lstm = LSTM(self.embed.embedding_dim, hidden_size=hidden_size//2, num_layers=num_layers,
                         batch_first=True, bidirectional=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.w_omega_id = nn.Parameter(torch.Tensor(
            hidden_size, hidden_size))
        self.u_omega_id = nn.Parameter(torch.Tensor(hidden_size , 1))
        nn.init.uniform_(self.w_omega_id, -0.1, 0.1)
        nn.init.uniform_(self.u_omega_id, -0.1, 0.1)

    def forward(self, chars, seq_len):  # 这里的名称必须和DataSet中相应的field对应，比如之前我们DataSet中有chars，这里就必须为chars
        # chars:[batch_size, max_len]
        # seq_len: [batch_size, ]
        chars = self.embed(chars)
        outputs, _ = self.lstm(chars, seq_len)
        outputs = self.dropout_layer(outputs)

        #scored_id = self._attention(outputs, self.w_omega_id, self.u_omega_id)
        # feat = torch.sum(scored_id, dim=1)  # 加权求和
        outputs = self.fc(outputs)
        outputs, _ = torch.max(outputs, dim=1)


        return {'pred':outputs}  # [batch_size,], 返回值必须是dict类型，且预测值的key建议设为pred

    def _attention(self,x,w_p,u_p):
        # Attention过程
        u = torch.tanh(torch.matmul(x, w_p))
        # u形状是(batch_size, seq_len, 2 * num_hiddens)
        att = torch.matmul(u, u_p)
        # att形状是(batch_size, seq_len, 1)
        att_score = F.softmax(att, dim=1)
        # att_score形状仍为(batch_size, seq_len, 1)
        scored_x = x * att_score
        return scored_x