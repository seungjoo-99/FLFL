import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param, loss_fn
from config import cfg
import numpy as np


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equal_inout = (in_planes == out_planes)
        self.shortcut = (not self.equal_inout) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                             padding=0, bias=False) or None

    def forward(self, x):
        if not self.equal_inout:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equal_inout else x)))
        # if self.drop_rate > 0:
        #     out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        out = torch.add(x if self.equal_inout else self.shortcut(x), out)
        return out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate):
        super().__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, data_shape, num_classes, depth, widen_factor, drop_rate):
        super().__init__()
        num_down = int(min(math.log2(data_shape[1]), math.log2(data_shape[2]))) - 3
        hidden_size = [16]
        for i in range(num_down + 1):
            hidden_size.append(16 * (2 ** i) * widen_factor)
        n = ((depth - 1) / (num_down + 1) - 1) / 2
        block = BasicBlock
        blocks = []
        blocks.append(nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False))
        blocks.append(NetworkBlock(n, hidden_size[0], hidden_size[1], block, 1, drop_rate))
        for i in range(num_down):
            blocks.append(
                nn.Sequential(
                    NetworkBlock(n, hidden_size[i + 1], hidden_size[i + 2], block, 2, drop_rate),
                    nn.Dropout(p=drop_rate),
                )
            )
        # blocks.append(nn.BatchNorm2d(hidden_size[-1]))
        # blocks.append(nn.ReLU(inplace=True))
        # blocks.append(nn.AdaptiveAvgPool2d(1))
        # blocks.append(nn.Flatten())

        self.bn1 = nn.BatchNorm2d(hidden_size[-1])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(hidden_size[-1], num_classes)
        self.feature_dim = hidden_size[-1]

    def forward(self, x):
        feats = self.blocks(x)

        feats = self.relu(self.bn1(feats))
        feats = self.avgpool(feats)
        feats = self.flatten(feats)

        logits = self.classifier(feats)



        output = {'logits':logits, 'feats':feats}
        return output

def process_bn_layer(m, momentum, track_running_stats):
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = momentum
        m.track_running_stats = track_running_stats
        if track_running_stats:
            m.register_buffer('running_mean', torch.zeros(m.num_features, device=m.weight.device))
            m.register_buffer('running_var', torch.ones(m.num_features, device=m.weight.device))
            m.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=m.weight.device))
        else:
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None
    return m

def reset_bn_layer_stats(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_running_stats()
    return m

def process_dropout_layer(m, p):
    if isinstance(m, nn.Dropout):
        m.p = p
    return m

def change_dropout_layer(model, p=0.4):
    model.apply(lambda m: process_dropout_layer(m, p=p))
    return model


def wresnet(connector=False, momentum=None, track=False):
    data_shape = cfg['data']['img_shape']
    num_classes = cfg['data']['num_classes']
    depth = cfg['model']['depth']
    widen_factor = cfg['model']['widen_factor']
    drop_rate = cfg['model']['drop_rate']
    model = WideResNet(data_shape, num_classes, depth, widen_factor, drop_rate)
    model.apply(init_param)
    model.apply(lambda m: process_bn_layer(m, momentum=momentum, track_running_stats=track))
    return model
