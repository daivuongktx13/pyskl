import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from mmcv.runner import load_checkpoint

# from .Temporal_shift.cuda.shift import Shift
from ..builder import BACKBONES
from ...utils import Graph, cache_checkpoint
from .utils import unit_tcn
from .utils.shiftgcn import get_shift_indexes


from .Temporal_shift.cuda.shift import Shift

def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)

class AdaptiveGraphConstruction(nn.Module):
    def __init__(self, channels, A, coff_embedding=3, init_strategy = 'no_shift') -> None:
        super(AdaptiveGraphConstruction, self).__init__()

        inter_channels = channels // coff_embedding
        self.inter_c = inter_channels
        self.channels = channels

        V = A.size(1)

        if init_strategy == 'no_shift':
            A_init = torch.eye(V).unsqueeze(0)
        elif init_strategy == 'global_shift':
            A_init = torch.ones((1, V, V))

        self.A = nn.Parameter(A_init)

        self.alpha = nn.Parameter(torch.zeros(1))

        self.conv_a = nn.Conv2d(channels, inter_channels, kernel_size=1)
        self.conv_b = nn.Conv2d(channels, inter_channels, kernel_size=1)
        self.tan = nn.Tanh()
        self.softmax = nn.Softmax(-1)


    def forward(self, x):
        N, C, T, V = x.size()
        Q = self.conv_a(x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
        K = self.conv_b(x).view(N, self.inter_c * T, V)
        Graph_B = self.tan(torch.matmul(Q, K) / Q.size(-1))  # N V V
        Graph_C = self.softmax(self.A.repeat(N, 1, 1) + Graph_B * self.alpha)
        shift_indexs = get_shift_indexes(Graph_C, self.channels, 'cuda')
        return shift_indexs

class Shift_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Shift_tcn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        bn_init(self.bn2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.shift_in = Shift(channel=in_channels, stride=1, init_scale=1)
        self.shift_out = Shift(channel=out_channels, stride=stride, init_scale=1)

        self.temporal_linear = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.kaiming_normal(self.temporal_linear.weight, mode='fan_out')

    def forward(self, x):
        x = self.bn(x)
        # shift1
        x = self.shift_in(x)
        x = self.temporal_linear(x)
        x = self.relu(x)
        # shift2
        x = self.shift_out(x)
        x = self.bn2(x)
        return x


class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=3):
        super(Shift_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        V = A.size(1)


        self.shift_in = AdaptiveGraphConstruction(in_channels, A, init_strategy='no_shift')
        self.shift_out = AdaptiveGraphConstruction(out_channels, A, init_strategy='no_shift') 

        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0,math.sqrt(1.0/out_channels))

        self.Linear_bias = nn.Parameter(torch.zeros(1,1,out_channels,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Linear_bias, 0)

        self.Feature_Mask = nn.Parameter(torch.ones(1,V,in_channels, requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(V*out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)


    def forward(self, x0):
        n, c, t, v = x0.size()
        x = x0.permute(0,2,3,1).contiguous()
        # shift1
        shift_in_indexs = self.shift_in(x0)
        shift_in_indexs = shift_in_indexs.view(n, -1).repeat(t, 1)
        x = x.view(n*t,v*c)
        x = torch.gather(x, dim = 1, index=shift_in_indexs)
        x = x.view(n*t,v,c)
        x = x * (torch.tanh(self.Feature_Mask)+1)

        x = torch.einsum('nwc,cd->nwd', (x, self.Linear_weight)).contiguous() # nt,v,c
        x = x + self.Linear_bias

        x1 = x.view(n, t, v, -1).permute(0,3,1,2).contiguous()

        # shift2
        shift_out_indexs = self.shift_out(x1)
        shift_out_indexs = shift_out_indexs.view(n, -1).repeat(t, 1)
        x = x.view(n*t,-1) 
        x = torch.gather(x, dim = 1, index=shift_out_indexs)
        x = self.bn(x)
        x = x.view(n,t,v,self.out_channels).permute(0,3,1,2) # n,c,t,v

        x = x + self.down(x0)
        x = self.relu(x)
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = Shift_gcn(in_channels, out_channels, A)
        self.tcn1 = Shift_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)

@BACKBONES.register_module()
class SHIFTGCNV2(nn.Module):
    
    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='MVC',
                 ch_ratio=2,
                 num_person=2,  # * Only used when data_bn_type == 'MVC'
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)

        self.data_bn_type = data_bn_type

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        bn_init(self.data_bn, 1)
    
    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x):
        N, M, T, V, C = x.size()

        x = x.permute(0, 1, 3, 4, 2).contiguous() # N, M, T, V, C -> N, M, V, C, T
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))

        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        x = x.reshape((N, M) + x.shape[1:])
        return x
