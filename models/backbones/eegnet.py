import torch.nn as nn
from easydict import EasyDict
from models.backbones.layers import Conv2dWithConstraint


class EEGNet(nn.Module):
    def __init__(self, args: EasyDict, dropout_rate: float = 0.5):
        super(EEGNet, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(
                1, args.F1, kernel_size=[1, args.sfreq // 2], padding="same", bias=False
            ),
            nn.BatchNorm2d(args.F1),
            Conv2dWithConstraint(
                args.F1,
                args.D * args.F1,
                kernel_size=[args.num_channels, 1],
                groups=args.F1,
                bias=False,
            ),
            nn.BatchNorm2d(args.D * args.F1),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=[1, args.sfreq // 32],
                stride=[1, args.sfreq // 32],
            ),
            nn.Dropout2d(dropout_rate),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(
                args.D * args.F1,
                args.F2,
                kernel_size=[1, args.sfreq // 8],
                padding="same",
                groups=args.D * args.F1,
                bias=False,
            ),
            nn.Conv2d(args.F2, args.F2, 1, bias=False),
            nn.BatchNorm2d(args.F2),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=[1, args.sfreq // 16], stride=[1, args.sfreq // 16]
            ),
            nn.Dropout2d(dropout_rate),
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(args.output_dim, args.num_classes)

    def forward(self, x):
        x = self.block_1(x)
        h = self.block_2(x)

        x = self.flatten(h)
        x = self.linear(x)

        return x
