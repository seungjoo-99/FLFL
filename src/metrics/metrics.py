import torch
import torch.nn.functional as F
from config import cfg
from utils import recur
import numpy as np


def Accuracy(output, target, topk=1):
    with torch.no_grad():
        if target.dtype != torch.int64:
            target = (target.topk(1, 1, True, True)[1]).view(-1)
        if output.dim() == 1:
            acc = (output == target).float().mean().item() * 100
        else:
            batch_size = target.size(0)
            pred_k = output.topk(topk, 1, True, True)[1]
            correct_k = pred_k.eq(target.view(-1, 1).expand_as(pred_k)).float().sum()
            acc = (correct_k * (100.0 / batch_size)).item()
    return acc


def MAccuracy(output, target, mask, topk=1):
    mask = mask.bool()
    if torch.any(mask):
        output = output[mask]
        target = target[mask]
        acc = Accuracy(output, target, topk)
    else:
        acc = 0
    return acc


def LabelRatio(mask):
    with torch.no_grad():
        lr = mask.float().mean().item()
    return lr

def PseudoLabelHist(pseudo_lb, mask):
    with torch.no_grad():
        pseudo_lb = pseudo_lb[mask]
        hist = np.histogram(pseudo_lb.cpu().numpy(), bins=np.arange(cfg['num_classes'] + 1))
    return hist

def SimCLRLoss(simclr_loss):
    with torch.no_grad():
        loss = simclr_loss.item()
    return loss

def KDLoss(kd_loss):
    with torch.no_grad():
        loss = kd_loss.item()
    return loss

class Metric(object):
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.pivot, self.pivot_name, self.pivot_direction = self.make_pivot()
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'Accuracy': (lambda input, output: recur(Accuracy, output['logits'], input['targets'])),
                       'PAccuracy': (lambda input, output: recur(Accuracy, output['pseudo_lb'],
                                                                 input['targets'][input['lb_len']:])),
                       'MAccuracy': (lambda input, output: recur(MAccuracy, output['pseudo_lb'],
                                                                 input['targets'][input['lb_len']:],
                                                                 output['mask'])),
                        'SMAccuracy': (lambda input, output: recur(MAccuracy, output['pseudo_lb'],
                                                                    input['targets'][input['lb_len']:],
                                                                    output['strong_mask'])),
                        'Rho': (lambda input, output: output.get('rho', 0)),
                       'LabelRatio': (lambda input, output: recur(LabelRatio, output['mask'])),
                       'SLabelRatio': (lambda input, output: recur(LabelRatio, output['strong_mask'])),
                       'PseudoLabelHist': (lambda input, output: recur(PseudoLabelHist, output['pseudo_lb'], output['mask'])),
                       'SimCLRLoss': (lambda input, output: output['simclr_loss'].item()),
                       'KDLoss': (lambda input, output: recur(KDLoss, output['kd_loss'])),
                       'sam_tt': (lambda input, output: output['tt']),
                       'sam_tf': (lambda input, output: output['tf']),
                       'sam_ff': (lambda input, output: output['ff']),
                       'sam_ft': (lambda input, output: output['ft']),
                       'max_probs_w': (lambda input, output: output.get('max_probs_ulb_w', 0)),
                       'max_probs_w_hat': (lambda input, output: output.get('max_probs_ulb_w_hat', 0)),
                       'max_probs_s': (lambda input, output: output.get('max_probs_ulb_s', 0)),
                       'max_probs_s_hat': (lambda input, output: output.get('max_probs_ulb_s_hat', 0))}

    def make_pivot(self):
        if cfg['data']['name'] in ['mnist', 'fashionmnist', 'cifar10', 'cifar100', 'svhn', 'stl10', 'image_net']:
            pivot = -float('inf')
            pivot_direction = 'up'
            pivot_name = 'Accuracy'
        else:
            raise ValueError('Not valid data name')
        return pivot, pivot_name, pivot_direction

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation

    def compare(self, val):
        if self.pivot_direction == 'down':
            compared = self.pivot > val
        elif self.pivot_direction == 'up':
            compared = self.pivot < val
        else:
            raise ValueError('Not valid pivot direction')
        return compared

    def update(self, val):
        self.pivot = val
        return
