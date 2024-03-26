import torch
import torch.nn as nn
from mmcv.cnn import normal_init
import math
import random
import numpy as np

from ..builder import HEADS
from .simple_head import SimpleHead


class InstanceBank(nn.Module):
    def __init__(self,
            channel = 256,
            mem_size = 684,
            positive_num = 128,
            negative_num = 512,
            T = 1.0,
            class_num = 60):
        super().__init__()

        self.mem_size = mem_size
        self.positive_num = positive_num
        self.negative_num = negative_num
        self.T = T
        self.class_num = class_num

        self.Bank = nn.Parameter(
            torch.zeros((mem_size * class_num, channel)), requires_grad=False
        )
        self.bank_flag = nn.Parameter(
            torch.zeros(mem_size * class_num), requires_grad=False
            ) 
        self.roundrobin = [0] * class_num
        self.bank_label = torch.arange(class_num).repeat(mem_size)
        self.cross_entropy = nn.CrossEntropyLoss()

    def get_enqueue_indexs(self, label):
        indies = []
        label = label.cpu().detach().tolist()
        for i in label:
            self.roundrobin[i] = self.roundrobin[i] + 1 if self.roundrobin[i] < self.class_num - 1 else 0
            indies.append(self.roundrobin[i] + i * self.mem_size)
        return indies

    def forward(self, f_normed, label):
        N, _ = f_normed.size()
        indies = self.get_enqueue_indexs(label)
        self.Bank[indies] = f_normed.detach()
        self.bank_flag[indies] = 1

        all_pairs = torch.einsum('n c, m c -> n m', f_normed, self.Bank)
        bank_label = self.bank_label.to(label.device) # mem_size
        positive_mask = (label.view(N, 1) == bank_label.view(1, -1)).view(N, self.mem_size * self.class_num) # n mem_size * class_num
        negative_mask = (1-positive_mask.float())

        positive_mask = positive_mask * self.bank_flag
        negative_mask = negative_mask * self.bank_flag

        combined_pairs_list = []

        for i in range(N):
            if (positive_mask[i].sum(dim=-1) < self.positive_num) or (negative_mask[i].sum(dim=-1) < self.negative_num):
                continue
            positive_pairs = torch.masked_select(all_pairs[i], mask=positive_mask[i].bool()).view(-1)
            positive_pairs_hard = positive_pairs.sort(dim=-1, descending=False)[0][:self.positive_num].view(1, self.positive_num, 1)

            negative_pairs = torch.masked_select(all_pairs[i], mask=negative_mask[i].bool()).view(-1)
            negative_pairs_hard = negative_pairs.sort(dim=-1, descending=True)[0][:self.negative_num].view(1, 1, self.negative_num)\
                .expand(-1, self.positive_num, -1)

            idx = random.sample(list(range(len(negative_pairs))), k=self.negative_num)
            negative_pairs_random = negative_pairs[idx].view(1, 1, self.negative_num).expand(-1, self.positive_num, -1)

            combined_pairs_hard2hard = torch.cat([positive_pairs_hard, negative_pairs_hard], -1).view(self.positive_num, -1)
            combined_pairs_hard2random = torch.cat([positive_pairs_hard, negative_pairs_random], -1).view(self.positive_num, -1)
            combined_pairs = torch.cat([combined_pairs_hard2hard, combined_pairs_hard2random], 0)
            combined_pairs_list.append((combined_pairs))

        if len(combined_pairs_list) == 0:
            return torch.tensor(0.0, device = label.device)

        combined_pairs = torch.cat(combined_pairs_list, 0)
        combined_label = torch.zeros(combined_pairs.size(0), device=f_normed.device).long()
        loss = self.cross_entropy(combined_pairs/self.T, combined_label)

        return loss

class SemanticBank(nn.Module):
    def __init__(self, channel, class_num, warmup_step = 100, alpha = 0.85, T = 1.0):
        super().__init__()

        self.SemanBank = nn.Parameter(
            torch.zeros((class_num, channel)), requires_grad=False
        )
        self.cross_entropy = nn.CrossEntropyLoss()
        self.bank_label = torch.arange(class_num)

        self.warmup_step = warmup_step
        self.alpha = alpha
        self.step = 0
        self.class_num = class_num
        self.T = T

    def update(self, f_normed, label):
        self.SemanBank[label] = self.SemanBank[label] * self.alpha + f_normed.detach()
        

    def forward(self, f_normed, label):
        N, _ = f_normed.size()
        self.update(f_normed, label)
        self.step += 1
        if self.step <= self.warmup_step:
            return torch.tensor(0.0, device = label.device)
        
        all_pairs = torch.einsum('n c, m c -> n m', f_normed, self.SemanBank)
        bank_label = self.bank_label.to(label.device) # class_num
        positive_mask = (label.view(N, 1) == bank_label.view(1, -1)).view(N, self.class_num) # n class_num
        negative_mask = (1-positive_mask.float())
        combined_pairs_list = []
        for i in range(N):
            positive_pairs = torch.masked_select(all_pairs[i], mask=positive_mask[i].bool()).view(1, 1)
            negative_pairs = torch.masked_select(all_pairs[i], mask=negative_mask[i].bool())\
                .view(1, self.class_num - 1)

            combined_pairs = torch.cat([positive_pairs, negative_pairs], -1)
            combined_pairs_list.append(combined_pairs)

        combined_pairs = torch.cat(combined_pairs_list, 0)
        combined_label = torch.zeros(combined_pairs.size(0), device=f_normed.device).long()
        loss = self.cross_entropy(combined_pairs/self.T, combined_label)
        return loss

class InfoNCE(nn.Module):
    def __init__(self, 
            in_channels=3 * 25 * 25, 
            out_channels=256, 
            mem_size=684, 
            positive_num=128, 
            negative_num=512, 
            T=1.0, 
            class_num=60,
            warmup_step = 100,
            alpha = 0.85):

        super(InfoNCE, self).__init__()

        self.trans = nn.Linear(in_channels, out_channels)
        nn.init.normal_(self.trans.weight, 0, math.sqrt(2. / class_num))
        nn.init.zeros_(self.trans.bias)

        self.instanceLoss = InstanceBank(out_channels, mem_size, positive_num, negative_num, T, class_num)
        self.semanticLoss = SemanticBank(out_channels, class_num, warmup_step, alpha, T)
        

    def forward(self, f, label):
        # f: N KsN2 
        # label: N
        f = self.trans(f)
        f_norm = f.norm(dim=-1, p=2, keepdim=True)
        f_normed = f / f_norm
        instance_loss = self.instanceLoss(f_normed, label)
        semantic_loss = self.semanticLoss(f_normed, label)
        return instance_loss + semantic_loss



@HEADS.register_module()
class GCNGCLHead(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.,
                 init_std=0.01,
                 gcl_in_channels = 5 * 25 * 25,
                 gcl_warmup_steps = 1000,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GCN',
                         **kwargs)

        self.gclLoss = InfoNCE(in_channels=gcl_in_channels,
            class_num=num_classes,
            warmup_step=gcl_warmup_steps)

    def gcl_loss(self, graph, label):
        return self.gclLoss(graph, label)