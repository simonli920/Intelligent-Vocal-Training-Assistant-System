import torch
from torch import nn


class CRNN(nn.Module):
    def __init__(self, rnn_layer, num_classes=3):
        super(CRNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5,1),stride=(1,1)), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 12, kernel_size=(5,1),stride=(1,1)),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.dense = nn.Linear(self.hidden_size, num_classes)
        self.state = None

    def forward(self, img, state=None):
        # print(1,img.shape)
        img = torch.unsqueeze(img,dim=1)
        feature = self.conv(img)
        # print(2,feature.shape)
        feature = feature.view(feature.shape[0], -1, feature.shape[3])
        # print(3, feature.shape)
        feature = feature.permute(2, 0, 1)
        # print(3, feature.shape)
        Y, self.state = self.rnn(feature, state)  # 格式说明：_input：（图片宽，批量大小，图片高）
        # print('Y',Y.shape)
        Y = Y[-1]
        # print('Y', Y.shape)
        output = self.dense(Y)  # Y：（图片宽，批量大小，隐藏单元个数）
        # print('output',output.shape)
        return output

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, num_classes=3):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
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