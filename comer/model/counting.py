import torch
from torch import nn
import torch.nn.functional as F


class ChannelAtt(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class AuxiliaryCounting(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(AuxiliaryCounting, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.trans_layer = nn.Sequential(
            nn.Conv2d(self.in_channel, 512, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(512))
        self.channel_att = ChannelAtt(512, 16)
        self.pred_layer = nn.Sequential(
            nn.Conv2d(512, self.out_channel, kernel_size=1, bias=False),
            nn.Sigmoid())

    def forward(self, x, mask):
        b, c, h, w = x.size()
        x = self.trans_layer(x)  # B, 512, H, W
        x = self.channel_att(x)  # B, 512, H, W
        x = self.pred_layer(x)  # B, 111, H, W
        if mask is not None:
            x = x * mask
        x = x.view(b, self.out_channel, -1)
        x1 = torch.sum(x, dim=-1)
        return x1, x.view(b, self.out_channel, h, w)


def gen_counting_label(labels, channel):
    b, t = labels.size()
    device = labels.device
    counting_labels = torch.zeros((b, channel))
    for i in range(b):
        for j in range(t):
            k = labels[i][j]
            counting_labels[i][k] += 1
    return counting_labels.to(device)


def cnt_loss(cnt_pred, indices):
    cnt_pred1, cnt_pred2 = cnt_pred
    cnt_pred = (cnt_pred1 + cnt_pred2) / 2
    cnt_lbl = gen_counting_label(indices, 111)
    loss = F.smooth_l1_loss
    cnt_loss = loss(cnt_pred1, cnt_lbl) + loss(cnt_pred2, cnt_lbl) + loss(cnt_pred, cnt_lbl)
    return cnt_loss
