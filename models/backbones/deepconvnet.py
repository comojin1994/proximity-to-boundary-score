import math
import torch.nn as nn
from easydict import EasyDict
from models.backbones.layers import Conv2dWithConstraint


class DeepConvNet(nn.Module):
    def __init__(self, args: EasyDict, dropout_rate: float = 0.5):
        super(DeepConvNet, self).__init__()

        kernel_size = int(args.sfreq * 0.02)

        self.temporal = Conv2dWithConstraint(
            1,
            args.D,
            kernel_size=[1, kernel_size],
            padding="valid",
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

        self.block_1 = nn.Sequential(
            self.temporal,
            self.spatial,
            nn.BatchNorm2d(args.D),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2]),
            nn.Dropout2d(dropout_rate),
        )

        self.block_2 = nn.Sequential(
            Conv2dWithConstraint(
                args.D,
                args.D * 2,
                kernel_size=[1, kernel_size],
                padding="valid",
                max_norm=2.0,
                bias=False,
            ),
            nn.BatchNorm2d(args.D * 2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2]),
            nn.Dropout2d(dropout_rate),
        )

        self.block_3 = nn.Sequential(
            Conv2dWithConstraint(
                args.D * 2,
                args.D * 4,
                kernel_size=[1, kernel_size],
                padding="valid",
                max_norm=2.0,
                bias=False,
            ),
            nn.BatchNorm2d(args.D * 4),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2]),
            nn.Dropout2d(dropout_rate),
        )

        self.block_4 = nn.Sequential(
            Conv2dWithConstraint(
                args.D * 4,
                args.D * 8,
                kernel_size=[1, kernel_size],
                padding="valid",
                max_norm=2.0,
                bias=False,
            ),
            nn.BatchNorm2d(args.D * 8),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2]),
            nn.Dropout2d(dropout_rate),
        )

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(args.output_dim, args.num_classes)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        h = self.block_4(x)

        x = self.flatten(h)
        x = self.dropout(x)
        x = self.linear(x)

        return x
