import torch.nn.functional as F
from torch import nn
import torch

class Block_plain(nn.Module):
    def __init__(self, in_ch, out_ch, up=False):
        super().__init__()
        self.up = up
        if up:
            self.conv1 = nn.Conv1d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose1d(out_ch, out_ch, 4, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv1d(out_ch, out_ch, 4, stride=2, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm1d(out_ch)
        self.bnorm2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply the first convolution
        x = self.conv1(x)
        x = self.bnorm1(self.relu(x))

        # Apply the second convolution
        x = self.conv2(x)
        x = self.bnorm2(self.relu(x))

        # Apply the transformation (upsample or downsample)
        x = self.transform(x)

        return x

class SimpleUnet_plain(nn.Module):
    def __init__(self, in_dim=1, dim=64, out_dim=1):
        super().__init__()

        in_channels = in_dim
        down_channels = (dim, 2 * dim, 4 * dim, 8 * dim, 16 * dim)
        up_channels = (16 * dim, 8 * dim, 4 * dim, 2 * dim, dim)

        self.conv0 = nn.Conv1d(in_channels, down_channels[0], 3, padding=1)
        self.downs = nn.ModuleList([
            Block_plain(down_channels[i], down_channels[i + 1])
            for i in range(len(down_channels) - 1)
        ])
        self.ups = nn.ModuleList([
            Block_plain(up_channels[i], up_channels[i + 1], up=True)
            for i in range(len(up_channels) - 1)
        ])
        self.output = nn.Conv1d(up_channels[-1], out_dim, 1)

    def forward(self, x, time=None):
        x = self.conv0(x)

        residual_inputs = []

        for down in self.downs:
            x = down(x)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()

            # if residual_x.shape[2] != x.shape[2]:
            #     diff = residual_x.shape[2] - x.shape[2]
            #     x = F.pad(x, (0, diff))

            x = torch.cat((x, residual_x), dim=1)
            x = up(x)

        output = self.output(x)
        return torch.tanh(output)


