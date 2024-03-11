import torch
import torch.nn as nn

from .init_func import bn_init, conv_init
from einops import rearrange, repeat

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(EdgeConv, self).__init__()
        
        self.k = k
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )

    def forward(self, x, dim=4): # N, C, T, V
        
        N, C, L = x.size()
        
        x = self.get_graph_feature(x, self.k)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        
        return x
        
    def knn(self, x, k):

        inner = -2 * torch.matmul(x.transpose(2, 1), x) # N, V, V
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = - xx - inner - xx.transpose(2, 1)
        
        idx = pairwise_distance.topk(k=k, dim=-1)[1] # N, V, k
        return idx
    
    def get_graph_feature(self, x, k, idx=None):
        N, C, V = x.size()
        if idx is None:
            idx = self.knn(x, k=k)
        device = x.get_device()
        
        idx_base = torch.arange(0, N, device=device).view(-1, 1, 1) * V
        
        idx = idx + idx_base
        idx = idx.view(-1)
        
        x = rearrange(x, 'n c v -> n v c')
        feature = rearrange(x, 'n v c -> (n v) c')[idx, :]
        feature = feature.view(N, V, k, C)
        x = repeat(x, 'n v c -> n v k c', k=k)
        
        feature = torch.cat((feature - x, x), dim=3)
        feature = rearrange(feature, 'n v k c -> n c v k')
        
        return feature
    
class AHA(nn.Module):
    def __init__(self, in_channels, num_layers):
        super(AHA, self).__init__()
        
        self.num_layers = num_layers
        
        inter_channels = in_channels // 4
                    
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        
        self.edge_conv = EdgeConv(inter_channels, inter_channels, k=3)
        
        self.aggregate = nn.Conv1d(inter_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        N, C, L, T, V = x.size()
        
        x_t = x.max(dim=-2, keepdim=False)[0]
        x_t = self.conv_down(x_t)

        x_sampled = x_t.mean(dim = -1)
        
        att = self.edge_conv(x_sampled, dim=3)
        att = self.aggregate(att).view(N, C, L, 1, 1)
        
        out = (x * self.sigmoid(att)).sum(dim=2, keepdim=False)
        
        return out