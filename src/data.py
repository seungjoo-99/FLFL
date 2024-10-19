import copy
import torch
import numpy as np
import models
from config import cfg
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset, SequentialSampler
from torch.utils.data.dataloader import default_collate
from utils import collate, to_device
from datasets import check_dataset_split
from math import ceil
import random


def get_dataset(data_dir='./data', include_lb_to_ulb=True):
    """
    create dataset

    Args
        args: argparse arguments
        algorithm: algorithm name, used for specific return items in __getitem__ of datasets
        dataset: dataset name 
        num_labels: number of labeled data in dataset
        num_classes: number of classes
        data_dir: data folder
        include_lb_to_ulb: flag of including labeled data into unlabeled data
    """
    from datasets import get_medmnist, get_cifar, get_svhn, get_stl10, get_imagenet

    if cfg['data']['name'] in ["cifar10", "cifar100"]:
        lb_dset, ulb_dset, eval_dset = get_cifar(data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
        test_dset = None
    elif cfg['data']['name'] == 'svhn':
        lb_dset, ulb_dset, eval_dset = get_svhn(data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
        test_dset = None
    elif cfg['data']['name'] == 'stl10':
        lb_dset, ulb_dset, eval_dset = get_stl10(data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
        test_dset = None
    elif cfg['data']['name'] in ["imagenet", "imagenet127"]:
        lb_dset, ulb_dset, eval_dset = get_imagenet(data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
        test_dset = None
    elif cfg['data']['name'] in ["tissuemnist"]:
        lb_dset, ulb_dset, eval_dset = get_medmnist(data_dir=data_dir,  include_lb_to_ulb=include_lb_to_ulb)
        test_dset = None
    else:
        return None
    
    dataset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset, 'test': test_dset}
    return dataset_dict

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_data_loader(dataset, only_eval=False, shuffle=True, train=True):
    batch_size = cfg['batch_size']
    ulb_batch_size = cfg['ulb_batch_size']
    eval_batch_size = cfg['eval_batch_size']
    ulb_ratio = cfg['ulb_ratio']
    data_loader = {}
    if not only_eval:
        lb_batch_size = batch_size
        ulb_batch_size = batch_size * ulb_ratio if train and check_dataset_split(dataset, 'train_lb') else ulb_batch_size if train else eval_batch_size
        num_batches = max(ceil(len(dataset['train_lb']) / lb_batch_size), ceil(len(dataset['train_ulb']) / (ulb_batch_size)))
        lb_num_samples = lb_batch_size * num_batches
        ulb_num_samples = ulb_batch_size * num_batches
            
        data_loader['train_lb'] = None if not check_dataset_split(dataset, 'train_lb') else DataLoader(
            dataset=dataset['train_lb'],
            batch_size=lb_batch_size,
            sampler=SequentialSampler(dataset['train_lb']) if not shuffle else RandomSampler(dataset['train_lb'], num_samples=lb_num_samples if cfg['setting'] !='lb-at-server' else len(dataset['train_lb'])),
            # drop_last=True,
            pin_memory=True,
            num_workers=cfg['num_workers'],
            collate_fn=None,
            worker_init_fn=seed_worker
        )
        data_loader['train_ulb'] = None if not check_dataset_split(dataset, 'train_ulb') else DataLoader(
            dataset=dataset['train_ulb'],
            batch_size=ulb_batch_size,
            sampler=SequentialSampler(dataset['train_ulb']) if not shuffle else RandomSampler(dataset['train_ulb'], num_samples=ulb_num_samples if cfg['setting'] !='lb-at-server' else len(dataset['train_ulb'])),
            # drop_last=True,
            pin_memory=True,
            num_workers=2 * cfg['num_workers'],
            collate_fn=None,
            worker_init_fn=seed_worker
        )
        data_loader['train_ulb_mix'] = None if not check_dataset_split(dataset, 'train_ulb_mix') else DataLoader(
            dataset=dataset['train_ulb_mix'],
            batch_size=ulb_batch_size,
            sampler=SequentialSampler(dataset['train_ulb_mix']) if not shuffle else RandomSampler(dataset['train_ulb_mix'], num_samples=ulb_num_samples if cfg['setting'] !='lb-at-server' else len(dataset['train_ulb_mix'])),
            # drop_last=True,
            pin_memory=True,
            num_workers=2 * cfg['num_workers'],
            collate_fn=None,
            worker_init_fn=seed_worker
        )
    data_loader['eval'] = None if not check_dataset_split(dataset, 'eval') else DataLoader(
        dataset=dataset['eval'],
        batch_size=eval_batch_size,
        pin_memory=True,
        num_workers=cfg['num_workers'],
        collate_fn=None,
        worker_init_fn=seed_worker
    )
    
    return data_loader

def iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    data_split, idx = {}, list(range(len(dataset)))
    for i in range(num_users):
        num_items_i = min(len(idx), num_items)
        data_split[i] = torch.tensor(idx)[torch.randperm(len(idx))[:num_items_i]].tolist()
        idx = list(set(idx) - set(data_split[i]))
    return data_split


def non_iid(dataset, num_users):
    target = torch.tensor(dataset.targets)
    data_split_mode_list = cfg['control_fl']['data_split_mode'].split('-')
    data_split_mode_tag = data_split_mode_list[-2]
    if data_split_mode_tag == 'l':
        data_split = {i: [] for i in range(num_users)}
        shard_per_user = int(data_split_mode_list[-1])
        target_idx_split = {}
        shard_per_class = int(shard_per_user * num_users / cfg['data']['num_classes'])
        for target_i in range(cfg['data']['num_classes']):
            target_idx = torch.where(target == target_i)[0]
            num_leftover = len(target_idx) % shard_per_class
            leftover = target_idx[-num_leftover:] if num_leftover > 0 else []
            new_target_idx = target_idx[:-num_leftover] if num_leftover > 0 else target_idx
            new_target_idx = new_target_idx.reshape((shard_per_class, -1)).tolist()
            for i, leftover_target_idx in enumerate(leftover):
                new_target_idx[i] = new_target_idx[i] + [leftover_target_idx.item()]
            target_idx_split[target_i] = new_target_idx
        target_split = list(range(cfg['data']['num_classes'])) * shard_per_class
        target_split = torch.tensor(target_split)[torch.randperm(len(target_split))].tolist()
        target_split = torch.tensor(target_split).reshape((num_users, -1)).tolist()
        for i in range(num_users):
            for target_i in target_split[i]:
                idx = torch.randint(len(target_idx_split[target_i]), (1,)).item()
                data_split[i].extend(target_idx_split[target_i].pop(idx))
    elif data_split_mode_tag == 'd':
        beta = float(data_split_mode_list[-1])
        dir = torch.distributions.dirichlet.Dirichlet(torch.tensor(beta).repeat(num_users))
        min_size = 0
        required_min_size = 10
        N = target.size(0)
        while min_size < required_min_size:
            data_split = [[] for _ in range(num_users)]
            for target_i in range(cfg['data']['num_classes']):
                target_idx = torch.where(target == target_i)[0]
                proportions = dir.sample()
                proportions = torch.tensor(
                    [p * (len(data_split_idx) < (N / num_users)) for p, data_split_idx in zip(proportions, data_split)])
                proportions = proportions / proportions.sum()
                split_idx = (torch.cumsum(proportions, dim=-1) * len(target_idx)).long().tolist()[:-1]
                split_idx = torch.tensor_split(target_idx, split_idx)
                data_split = [data_split_idx + idx.tolist() for data_split_idx, idx in zip(data_split, split_idx)]
            min_size = min([len(data_split_idx) for data_split_idx in data_split])
        data_split = {i: data_split[i] for i in range(num_users)}
    else:
        raise ValueError('Not valid data split mode tag')
    return data_split

def split_dataset(dataset, client_datasets_idx=None):
    if client_datasets_idx is None:
        # split labeled dataset
        num_users = cfg['control_fl']['num_clients']
        num_classes = cfg['data']['num_classes']
        num_labels = len(dataset['train_lb'].targets)
        lb_to_idx = np.array([np.where(dataset['train_lb'].targets == c)[0] for c in range(num_classes)])
        if cfg['setting'] == 'lb-at-server':
            idx_lb = [[] for _ in range(num_users)]
        elif cfg['setting'] == 'lb-at-clients':
            assert num_labels % (num_users * num_classes) == 0
            num_labels_per_client_class = int(num_labels / (num_users * num_classes))
            idx_lb = [lb_to_idx[:, i * num_labels_per_client_class: (i + 1) * num_labels_per_client_class].flatten().tolist() for i in range(num_users)]
        idx_ulb = iid(dataset['train_ulb'], num_users) if cfg['control_fl']['data_split_mode'] == 'iid' else non_iid(dataset['train_ulb'], num_users)
        client_datasets_idx = [{'train_lb':idx_lb[i], 'train_ulb': idx_ulb[i]} for i in range(num_users)]
    client_datasets = []
    for i in range(num_users):
        client_dataset = {}
        client_dataset['train_lb'] = Subset(dataset['train_lb'], idx_lb[i])
        client_dataset['train_ulb'] = Subset(dataset['train_ulb'], idx_ulb[i])
        client_datasets.append(client_dataset)
    return client_datasets, client_datasets_idx

def make_sbn_stats(dataset, model):
    # reset running stats of BNs layer
    model.apply(models.reset_bn_layer_stats)
    # temporarily remove augmentation from train data
    train_lb_transform = dataset['train_lb'].transform
    train_ulb_transform = dataset['train_ulb'].transform
    dataset['train_lb'].transform = dataset['eval'].transform
    dataset['train_ulb'].transform = dataset['eval'].transform
    with torch.no_grad():
        data_loader = make_data_loader(dataset, shuffle=False, train=False)
        model.train()
        if cfg['control_fl']['sbn'] > 0:
            for data_lb in data_loader['train_lb']:
                data_lb = to_device(data_lb, cfg['device'])
                model(data_lb['x_lb'])
            if cfg['control_fl']['sbn'] > 1:
                for data_ulb in data_loader['train_ulb']:
                    data_ulb = to_device(data_ulb, cfg['device'])
                    model(data_ulb['x_ulb_w'])
    dataset['train_lb'].transform = train_lb_transform
    dataset['train_ulb'].transform = train_ulb_transform
    return model
