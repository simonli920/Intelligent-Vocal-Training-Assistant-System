from torch import nn


class GRU(nn.Module):
    def __init__(self, input_size=256, hidden_size=1024, num_classes=3):
        super(GRU, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=False)  # 定义rnn块
        self.hidden_size = self.rnn.hidden_size * (2 if self.rnn.bidirectional else 1)
        self.dense = nn.Linear(self.hidden_size, num_classes)
        self.state = None

    def forward(self, inputs, state=None): # inputs: (batch, seq_len)
        # print('inputs',inputs.shape)
        _inputs = inputs.permute(2, 0, 1)  # 格式说明：input：（批量大小，图片高，图片宽）
        # print('_inputs',_inputs.shape)
        Y, self.state = self.rnn(_inputs, state)       # 格式说明：_input：（图片宽，批量大小，图片高）
        # print('Y',Y.shape)
        Y = Y[-1]
        # print(torch.equal(Y, self.state[0][0]))
        output = self.dense(Y)                                     # Y：（图片宽，批量大小，隐藏单元个数）
        # print('output',output.shape)
        return output                                     # 取图片的最后一列预测，output：（批量大小，类别数目）
                                                          # state：（批量大小，rnn块传递参数个数）
