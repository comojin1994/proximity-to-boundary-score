import torch.nn as nn
import torch
from easydict import EasyDict
from models.backbones.layers import Conv2dWithConstraint


class ShallowConvNet(nn.Module):
    def __init__(self, args: EasyDict, dropout_rate: float = 0.5):
        super(ShallowConvNet, self).__init__()

        kernel_size = int(args.sfreq * 0.12)
        pooling_kernel_size = int(args.sfreq * args.pooling_size)
        pooling_stride_size = int(args.sfreq * args.pooling_size * (1 - args.hop_size))

        self.temporal = Conv2dWithConstraint(
            1,
            args.D,
            kernel_size=[1, kernel_size],
            padding="same",
            max_norm=2.0,
            bias=False,
        )
        self.spatial = Conv2dWithConstraint(
            args.D,
            args.D,
            kernel_size=[args.num_channels, 1],
            padding="valid",
            max_norm=2.0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(args.D)

        self.avg_pool = nn.AvgPool2d(
            kernel_size=[1, pooling_kernel_size], stride=[1, pooling_stride_size]
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(args.output_dim, args.num_classes)

    def forward(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = torch.square(x)
        x = self.avg_pool(x)
        x = torch.log(torch.clamp(x, min=1e-06))
        x = self.bn(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x
