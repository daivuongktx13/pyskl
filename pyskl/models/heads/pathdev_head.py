import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmcv import dump

from ..builder import HEADS
from .base import BaseHead

import sys
sys.path.insert(0, '/home/trungct/pyskl/DevNet')
from development.nn import development_layer
from development.so import so
from development.se import se

@HEADS.register_module()
class PathDevHead(BaseHead):
    """ A simple classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.0,
                 init_std=0.01,
                 lie_hidden_size=16,
                 dev_lie = 'se',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.dev_lie = se if dev_lie == 'se' else so

        self.path_dev = development_layer(
            input_size=in_channels, hidden_size=lie_hidden_size, param=self.dev_lie,
            return_sequence=False)

        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.in_c, num_classes)

        self.zs = []
        self.count = 0

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        pool = nn.AdaptiveAvgPool1d(1)
        N, M, C, T, V = x.shape
        x = x.reshape(N * M, C, T, V)
        x = x.permute(0, 3, 2, 1).contiguous().view(N * M * V, T, C)
        x = self.path_dev(x)

        x = x.view(N * M, V, C).permute(0, 2, 1)

        x = pool(x)
        x = x.reshape(N, M, C)
        x = x.mean(dim=1)

        assert x.shape[1] == self.in_c
        zs = x
        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc_cls(x)
        return cls_score, zs

@HEADS.register_module()
class PathDevHead2(BaseHead):
    """ A simple classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.0,
                 init_std=0.01,
                 lie_hidden_size=16,
                 dev_lie = 'se',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.dev_lie = se if dev_lie == 'se' else so

        self.path_dev = development_layer(
            input_size=in_channels, hidden_size=lie_hidden_size, param=self.dev_lie,
            return_sequence=False)

        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.in_c, num_classes)

        self.zs = []
        self.count = 0

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        N, M, C, T, V = x.shape

        x = x.reshape(N * M, C, T, V).mean(3)
        x = x.permute(0, 2, 1).contiguous()
        x = self.path_dev(x)

        x = x.reshape(N, M, C)
        x = x.mean(dim=1)

        assert x.shape[1] == self.in_c
        zs = x
        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc_cls(x)
        return cls_score, zs