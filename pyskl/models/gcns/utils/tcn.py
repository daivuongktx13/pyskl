import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer

from .init_func import bn_init, conv_init
from .dev_layer import *


class unit_tcn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, norm='BN', dropout=0):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1] if norm is not None else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))

    def init_weights(self):
        conv_init(self.conv)
        bn_init(self.bn, 1)


class mstcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 dropout=0.,
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                 stride=1):

        super().__init__()
        # Multiple branches of temporal convolution
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=None))
            branches.append(branch)

        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels), self.act, nn.Conv2d(tin_channels, out_channels, kernel_size=1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x):
        N, C, T, V = x.shape

        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        feat = torch.cat(branch_outs, dim=1)
        feat = self.transform(feat)
        return feat

    def forward(self, x):
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)

    def init_weights(self):
        pass


class dgmstcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 num_joints=25,
                 dropout=0.,
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                 stride=1):

        super().__init__()
        # Multiple branches of temporal convolution
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act = nn.ReLU()
        self.num_joints = num_joints
        # the size of add_coeff can be smaller than the actual num_joints
        self.add_coeff = nn.Parameter(torch.zeros(self.num_joints))

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=None))
            branches.append(branch)

        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels), self.act, nn.Conv2d(tin_channels, out_channels, kernel_size=1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x):
        N, C, T, V = x.shape
        x = torch.cat([x, x.mean(-1, keepdim=True)], -1)

        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        local_feat = out[..., :V]
        global_feat = out[..., V]
        global_feat = torch.einsum('nct,v->nctv', global_feat, self.add_coeff[:V])
        feat = local_feat + global_feat

        feat = self.transform(feat)
        return feat

    def forward(self, x):
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)

# class DevLSTM(nn.Module):

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  mid_channels=None,
#                  dev_lie = 'se',
#                  lie_hidden_size=10,
#                  dev_dilation = [1, 2],
#                  dropout=0.,
#                  stride=1):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.dev_kernel = 2 if stride == 1 else 3
#         self.dev_lie = se if dev_lie == 'se' else so

#         num_dev_branch = len(dev_dilation)

#         self.dev_layers = []
#         use_sp = True
#         for i in range(num_dev_branch):
#             if stride == 1 and i != 0:
#                 use_sp = False
#                 num_dev_branch = 1

#             self.dev_layers.append(dilation_dev(
#                     dilation=dev_dilation[i],
#                     h_size=lie_hidden_size,
#                     param=self.dev_lie,
#                     input_channel=out_channels,
#                     kernel_size=self.dev_kernel,
#                     stride=stride,
#                     return_sequence=False,
#                     use_sp=use_sp))

#         channel_num = int(num_dev_branch * out_channels + len(dev_dilation) * lie_hidden_size * lie_hidden_size) 

#         self.dev_layers = nn.ModuleList(self.dev_layers)

#         self.lstm = nn.LSTM(
#             input_size=channel_num,
#             hidden_size=out_channels,
#             num_layers=1,
#             batch_first=True,
#             bidirectional=False
#         )

#         self.pooling = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
#             nn.BatchNorm2d(out_channels)  
#         )

#         self.conv = nn.Sequential(nn.Conv2d(out_channels, int(out_channels/2), kernel_size=1, padding=0),
#             nn.BatchNorm2d(int(out_channels/2)),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(int(out_channels/2), (out_channels), 
#                 kernel_size=(5, 1),
#                 padding=(2, 0),
#                 stride=(stride, 1)),
#             nn.BatchNorm2d(int(out_channels))
#         )

#         self.bn = nn.BatchNorm1d(out_channels)
#         self.gelu = torch.nn.GELU()

#     def forward(self, x):
#         N_M, C, T, V = x.shape

#         x_pooling = self.pooling(x)
#         x_conv = self.conv(x)

#         x = x.permute(0, 3, 2, 1).contiguous().view(N_M * V, T, self.out_channels).contiguous()

#         x_dev = []
#         for dev_layer in self.dev_layers:
#             x_dev.append(dev_layer(x).type_as(x))
#         x_dev = torch.cat(x_dev, axis=-1)

#         _, T_segments, _ = x_dev.shape

#         x, _ = self.lstm(x_dev)
#         x = self.bn(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
#         x = x.view(N_M, V, T_segments, self.out_channels).permute(0, 3, 2, 1).contiguous()

#         x = x_pooling + x_conv + x 
#         out = self.gelu(x)
#         return out
    
#     def init_weights(self):
#         pass

class MyDevLSTM(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 dev_lie = 'se',
                 lie_hidden_size=10,
                 dev_dilation = [1],
                 dropout=0.,
                 stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dev_kernel = 2 if stride == 1 else 3
        self.dev_lie = se if dev_lie == 'se' else so

        num_dev_branch = len(dev_dilation)

        mid_channels = out_channels // 3
        rem_mid_channels = out_channels - mid_channels * 2

        self.dev_layers = []
        use_sp = True
        for i in range(num_dev_branch):
            if stride == 1 and i != 0:
                use_sp = False
                num_dev_branch = 1

            self.dev_layers.append(dilation_dev(
                    dilation=dev_dilation[i],
                    h_size=lie_hidden_size,
                    param=self.dev_lie,
                    input_channel=mid_channels,
                    kernel_size=self.dev_kernel,
                    stride=stride,
                    return_sequence=False,
                    use_sp=use_sp))

        channel_num = int(num_dev_branch * mid_channels + len(dev_dilation) * lie_hidden_size * lie_hidden_size) 

        self.dev_layers = nn.ModuleList(self.dev_layers)

        self.down = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.lstm = nn.LSTM(
            input_size=channel_num,
            hidden_size=mid_channels,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.pooling = nn.Sequential(
            nn.Conv2d(in_channels, rem_mid_channels, kernel_size=1),
            nn.BatchNorm2d(rem_mid_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1,0))  
        )

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 
                kernel_size=(5, 1),
                padding=(2, 0),
                stride=(stride, 1))
        )
        tin_channels = mid_channels * (3 - 1) + rem_mid_channels
        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels), nn.ReLU(inplace=True), nn.Conv2d(tin_channels, out_channels, kernel_size=1))

        self.gelu = torch.nn.GELU()

    def forward(self, x):
        N_M, C, T, V = x.shape

        x_pooling = self.pooling(x)
        x_conv = self.conv(x)

        x = self.down(x)

        x = x.permute(0, 3, 2, 1).contiguous().view(N_M * V, T, -1).contiguous()

        x_dev = []
        for dev_layer in self.dev_layers:
            x_dev.append(dev_layer(x).type_as(x))
        x_dev = torch.cat(x_dev, axis=-1)

        _, T_segments, _ = x_dev.shape

        x, _ = self.lstm(x_dev)
        x = x.permute(0, 2, 1).contiguous().permute(0, 2, 1).contiguous()
        x = x.view(N_M, V, T_segments, -1).permute(0, 3, 2, 1).contiguous()

        x = torch.cat([x_pooling, x_conv, x], dim = 1)
        x = self.transform(x)
        out = self.gelu(x)
        return out
    
    def init_weights(self):
        pass

class devmstcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 dropout=0.,
                 ms_cfg=[(3, 1), (3, 2), ('max', 3), '1x1'],
                 dev_dilation = [1],
                 dev_lie = 'se',
                 lie_hidden_size=10,
                 stride=1):

        super().__init__()
        # Multiple branches of temporal convolution
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg) + 1
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=None))
            branches.append(branch)
        
        ### Dev LSTM part
        num_dev_branch = len(dev_dilation)
        self.dev_lie = se if dev_lie == 'se' else so
        self.dev_kernel = 2 if stride == 1 else 3
        self.dev_layers = []
        use_sp = True
        for i in range(num_dev_branch):
            if stride == 1 and i != 0:
                use_sp = False
                num_dev_branch = 1

            self.dev_layers.append(dilation_dev(
                    dilation=dev_dilation[i],
                    h_size=lie_hidden_size,
                    param=self.dev_lie,
                    input_channel=mid_channels,
                    kernel_size=self.dev_kernel,
                    stride=stride,
                    return_sequence=False,
                    use_sp=use_sp))

        channel_num = int(num_dev_branch * mid_channels + len(dev_dilation) * lie_hidden_size * lie_hidden_size) 

        self.dev_layers = nn.ModuleList(self.dev_layers)

        self.down = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.lstm = nn.LSTM(
            input_size=channel_num,
            hidden_size=mid_channels,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        #####

        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels), self.act, nn.Conv2d(tin_channels, out_channels, kernel_size=1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x):
        N, C, T, V = x.shape

        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        
        ### Dev LSTM
        x = self.down(x)
        x = x.permute(0, 3, 2, 1).contiguous().view(N * V, T, -1).contiguous()

        x_dev = []
        for dev_layer in self.dev_layers:
            x_dev.append(dev_layer(x).type_as(x))
        x_dev = torch.cat(x_dev, axis=-1)

        _, T_segments, _ = x_dev.shape

        x, _ = self.lstm(x_dev)
        x = x.permute(0, 2, 1).contiguous().permute(0, 2, 1).contiguous()
        x = x.view(N, V, T_segments, -1).permute(0, 3, 2, 1).contiguous()
        branch_outs.append(x)
        ### 
        
        feat = torch.cat(branch_outs, dim=1)
        feat = self.transform(feat)
        return feat

    def forward(self, x):
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)

    def init_weights(self):
        pass