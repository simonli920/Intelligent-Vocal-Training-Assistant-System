import torch.nn as nn
import torch


class FTB(nn.Module):
    def __init__(self, input_dim=257, in_channel=9, r_channel=5):
        super(FTB, self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, r_channel, kernel_size=[1, 1]),
            nn.BatchNorm2d(r_channel),
            nn.ReLU()
        )

        self.conv1d = nn.Sequential(
            nn.Conv1d(r_channel * input_dim, in_channel, kernel_size=9, padding=4),
            nn.BatchNorm1d(in_channel),
            nn.ReLU()
        )
        self.freq_fc = nn.Linear(input_dim, input_dim, bias=False)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=[1, 1]),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

    def forward(self, inputs):
        '''
        inputs should be [Batch, Ca, Dim, Time]
        '''
        # T-F attention
        conv1_out = self.conv1(inputs)
        B, C, D, T = conv1_out.size()
        reshape1_out = torch.reshape(conv1_out, [B, C * D, T])
        conv1d_out = self.conv1d(reshape1_out)
        conv1d_out = torch.reshape(conv1d_out, [B, self.in_channel, 1, T])

        # now is also [B,C,D,T]
        att_out = conv1d_out * inputs

        # tranpose to [B,C,T,D]
        att_out = torch.transpose(att_out, 2, 3)
        freqfc_out = self.freq_fc(att_out)
        att_out = torch.transpose(freqfc_out, 2, 3)

        cat_out = torch.cat([att_out, inputs], 1)
        outputs = self.conv2(cat_out)
        return outputs


class TSB(nn.Module):
    def __init__(self, input_dim=257, channel_amp=9, channel_phase=8):
        super(TSB, self).__init__()
        self.ftb1 = FTB(input_dim=input_dim, in_channel=channel_amp)
        self.amp_conv1 = nn.Sequential(
            nn.Conv2d(channel_amp, channel_amp, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU()
        )
        self.amp_conv2 = nn.Sequential(
            nn.Conv2d(channel_amp, channel_amp, kernel_size=(1, 25), padding=(0, 12)),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU()
        )
        self.amp_conv3 = nn.Sequential(
            nn.Conv2d(channel_amp, channel_amp, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU()
        )
        self.ftb2 = FTB(input_dim=input_dim, in_channel=channel_amp)

    def forward(self, amp):
        '''
        amp should be [Batch, Cr, Dim, Time]
        '''
        amp_out1 = self.ftb1(amp)
        amp_out2 = self.amp_conv1(amp_out1)
        amp_out3 = self.amp_conv2(amp_out2)
        amp_out4 = self.amp_conv3(amp_out3)
        amp_out5 = self.ftb2(amp_out4)
        return amp_out5


class LCZNet(nn.Module):
    def __init__(self, num_classes=3, channel_amp=12, num_blocks=3, rnn_nums=300):
        super(LCZNet, self).__init__()
        self.num_classes = num_classes
        self.channel_amp = channel_amp
        self.num_blocks = num_blocks
        self.feat_dim = 112
        self.rnn_nums = rnn_nums
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, channel_amp, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU(),
            nn.Conv2d(channel_amp, channel_amp, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU()
        )
        self.tsbs = nn.ModuleList()
        for idx in range(self.num_blocks):
            self.tsbs.append(
                TSB(input_dim=self.feat_dim, channel_amp=channel_amp)
            )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(channel_amp, 8, kernel_size=[1, 1]),
                        nn.BatchNorm2d(8),
                        nn.ReLU(),
                    )
        self.fcs = nn.Sequential(
                    nn.Linear(rnn_nums,100),
                    nn.ReLU(),
                    nn.Linear(100,33),
                    nn.ReLU(),
                    nn.Linear(33,num_classes),
                    nn.Sigmoid()
                )
        self.rnn = nn.GRU(
                        self.feat_dim * 8,
                        rnn_nums,
                        bidirectional=False
                    )

    def forward(self, img):
        # print('input', img.shape)
        img = torch.unsqueeze(img,dim=1)
        # print('input2', img.shape)
        f1 = self.conv1(img)
        # print('f1', f1.shape)

        f1_cp = f1
        for idx, layer in enumerate(self.tsbs):
            if idx != 0:
                f2 += f1_cp
            f2 = layer(f1)

        # f2 = self.tsbs(f1)
        # print('f2', f2.shape)

        # f3 = self.conv2(f2)
        # print('f3', f3.shape)

        # 改变f3的形状，使之能输入到rnn中
        # B, C, D, T = f3.size()
        # f3 = f3.permute(3,0,1,2)
        # print('f3', f3.shape)
        # T, B, C, D = f3.size()
        # f3=  torch.reshape(f3, [T,B,C*D])
        # print('f3', f3.shape)

        # all_out, f4 = self.rnn(f3)
        # print('all_out', all_out.shape)
        # print('f4', f4.shape)
        # print('f4_features', f4.view(img.shape[0], -1).shape)

        f5 = self.fcs(f2.view(img.shape[0], -1))
        # print('f5', f5.shape)
        return f5