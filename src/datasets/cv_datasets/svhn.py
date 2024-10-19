# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math

from torchvision import transforms
from .datasetbase import BasicDataset
from ..utils import split_ssl_data
from ..augmentation import RandAugment, RandomResizedCropAndInterpolation
from config import cfg

mean, std = {}, {}
mean['svhn'] = [0.4380, 0.4440, 0.4730]
std['svhn'] = [0.1751, 0.1771, 0.1744]
img_size = 32

def get_transform(mean, std, crop_size, train=True, crop_ratio=0.95):
    img_size = int(crop_size / crop_ratio)

    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.Resize(img_size),
                                   transforms.RandomCrop(crop_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.Resize(crop_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])


def get_svhn(data_dir='./data', include_lb_to_ulb=True):

    name = cfg['data']['name'].lower()

    crop_size = cfg['data']['img_shape'][1]
    crop_ratio = cfg['data']['crop_ratio']
    num_classes = cfg['data']['num_classes']
    num_labels = cfg['control_ssl']['num_labels']
    alg = cfg['control_ssl']['algorithm']

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        # RandomResizedCropAndInterpolation(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        # RandomResizedCropAndInterpolation((crop_size, crop_size), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        # transforms.Resize(img_size),
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name],)
    ])


    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())
    dset_base = dset(data_dir, split='train', download=True)
    data, targets = dset_base.data.transpose([0, 2, 3, 1]), dset_base.labels
    # data_b, targets_b = dset_base.data.transpose([0, 2, 3, 1]), dset_base.labels
    # dset_extra = dset(data_dir, split='extra', download=True)
    # data_e, targets_e = dset_extra.data.transpose([0, 2, 3, 1]), dset_extra.labels
    # data = np.concatenate([data_b, data_e])
    # targets = np.concatenate([targets_b, targets_e])
    # del data_b, data_e
    # del targets_b, targets_e
    

    if alg == 'fullysupervised' or num_labels == -1:
        lb_data = np.array(data)
        lb_targets = np.array(targets)
        ulb_data = []
        ulb_targets = []
        cfg['control_ssl']['num_labels'] = len(data)
    else:
        lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(data, targets, num_classes, 
                                                                    lb_num_labels=num_labels,
                                                                    include_lb_to_ulb=include_lb_to_ulb)
    
    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong, False)

    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)

    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, split='test', download=True)
    data, targets = dset.data.transpose([0, 2, 3, 1]), dset.labels
    eval_dset = BasicDataset(alg, data, targets, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset
