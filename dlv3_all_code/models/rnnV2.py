import numpy as np
import torch
from torch import nn, optim
import torch


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 定义运行设备，显卡可用就用显卡


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, inputs, state=None):
        print((inputs.size()))
        Y, state = self.rnn(inputs, state)
        if isinstance(inputs, torch.nn.utils.rnn.PackedSequence):
            print((Y.data.size()))
            output = self.dense(Y.data)
        else:
            output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, state






