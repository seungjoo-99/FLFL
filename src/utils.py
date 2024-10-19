import collections.abc as container_abcs
import errno
import numpy as np
import os
import pickle
import torch
import torch.optim as optim
from itertools import repeat
from torchvision.utils import save_image
from config import cfg
import models

def select_clients(num_clients, active_rate, num_rounds):
    selected = []
    num_active_clients = int(np.ceil(active_rate * num_clients))

    for i in range(num_rounds):
        client_id = torch.arange(num_clients)[torch.randperm(num_clients)[:num_active_clients]].tolist()
        selected.append(client_id)
    
    return selected



def check_algorithm(target):
    if target in cfg['control_ssl']['algorithm']:
        return True
    return False

def get_model(track=False, to_device=True):
    kwargs = {}
    kwargs['track'] = track

    model = eval('models.{}(**kwargs)'.format(cfg['model']['type']))

    if to_device:
        model.to(cfg["device"])
    return model


def eval_only(model_path):
    result = resume(cfg['model_tag'], model_path, load_tag='best')
    epoch = result['epoch'] - 1
    logger = result['logger']
    info = {'info': ['TESTING', 'Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}'.format(epoch)]}
    # logger.append(info, 'test', mean=False)
    print(' | '.join(info['info']))
    # logger.append(info, 'train', mean=False)
    print('\tLoss: {:.2f}\n\tAccuracy: {:.2f}%'.format(logger.mean['test/Loss'], logger.mean['test/Accuracy']))

def check_exists(path):
    return os.path.exists(path)

def zip_data_loader(data_loader):
    if data_loader["train_lb"] is None:
        data_loader["train_lb"] = [None] * len(data_loader["train_ulb"])
    if data_loader["train_ulb"] is None:
        data_loader["train_ulb"] = [None] * len(data_loader["train_lb"])
        if cfg['control_ssl']['mixup'] > 0: 
            data_loader["train_ulb_mix"] = [None] * len(data_loader["train_lb"])
    if cfg['control_ssl']['mixup'] > 0: 
        dl_zipped = zip(data_loader["train_lb"], data_loader["train_ulb"], data_loader['train_ulb_mix'])
    else:
        dl_zipped = zip(data_loader["train_lb"], data_loader["train_ulb"])
    return dl_zipped

def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


def save(input, path, mode='torch'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path)
    elif mode == 'np':
        np.save(path, input, allow_pickle=True)
    elif mode == 'pickle':
        pickle.dump(input, open(path, 'wb'))
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'np':
        return np.load(path, allow_pickle=True)
    elif mode == 'pickle':
        return pickle.load(open(path, 'rb'))
    else:
        raise ValueError('Not valid save mode')
    return


def save_img(img, path, nrow=10, padding=2, pad_value=0, range=None):
    makedir_exist_ok(os.path.dirname(path))
    normalize = False if range is None else True
    save_image(img, path, nrow=nrow, padding=padding, pad_value=pad_value, normalize=normalize, range=range)
    return


def to_device(input, device):
    output = recur(lambda x, y: x.to(y), input, device)
    return output


def ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


def apply_fn(module, fn):
    for n, m in module.named_children():
        if hasattr(m, fn):
            exec('m.{0}()'.format(fn))
        if sum(1 for _ in m.named_children()) != 0:
            exec('apply_fn(m,\'{0}\')'.format(fn))
    return

def metric_batch(data_batch=None, data_lb=None, data_ulb=None):
    in_dict = {'lb_len': 0}
    if data_batch is not None:
        data_lb, data_ulb = data_batch[:2]
    if data_ulb is None:
        in_dict['targets'] = data_lb['y_lb']
        in_dict['lb_len'] = len(data_lb['y_lb'])
    elif data_lb is None:
        in_dict['targets'] = data_ulb['y_ulb'] 
    else:
        in_dict['targets'] = torch.cat([data_lb['y_lb'], data_ulb['y_ulb']])
        in_dict['lb_len'] = len(data_lb['y_lb'])
    return in_dict

def recur(fn, input, *args):
    if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        output = fn(input, *args)
    elif isinstance(input, list):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
    elif isinstance(input, tuple):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
        output = tuple(output)
    elif isinstance(input, dict):
        output = {}
        for key in input:
            if isinstance(key, str) and 'idx' in key:
                output[key] = input[key].tolist()
            else:
                output[key] = recur(fn, input[key], *args)
    elif isinstance(input, str):
        output = input
    elif input is None:
        output = None
    else:
        raise ValueError('Not valid input type')
    return output

def process_control():
    cfg['control_ssl']['algorithm'] = cfg['control_ssl']['algorithm'].lower()
    # Hack for now: mixup is only used when global_pseudo_lb is 2
    if cfg['control_fl']['global_pseudo_lb'] != 2:
       cfg['control_ssl']['mixup'] = 0
    # End of hack
    # Hack for now: maskout_ulb is only used when global_pseudo_lb is 2
    if cfg['control_fl']['global_pseudo_lb'] == 2:
       cfg['maskout_ulb'] = False
    # End of hack
    return


def make_stats():
    stats = {}
    stats_path = './res/stats'
    makedir_exist_ok(stats_path)
    filenames = os.listdir(stats_path)
    for filename in filenames:
        stats_name = os.path.splitext(filename)[0]
        stats[stats_name] = load(os.path.join(stats_path, filename))
    return stats


class Stats(object):
    def __init__(self, dim):
        self.dim = dim
        self.n_samples = 0
        self.n_features = None
        self.mean = None
        self.std = None

    def update(self, data):
        data = data.transpose(self.dim, -1).reshape(-1, data.size(self.dim))
        if self.n_samples == 0:
            self.n_samples = data.size(0)
            self.n_features = data.size(1)
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
        else:
            m = float(self.n_samples)
            n = data.size(0)
            new_mean = data.mean(dim=0)
            new_std = 0 if n == 1 else data.std(dim=0)
            old_mean = self.mean
            old_std = self.std
            self.mean = m / (m + n) * old_mean + n / (m + n) * new_mean
            self.std = torch.sqrt(m / (m + n) * old_std ** 2 + n / (m + n) * new_std ** 2 + m * n / (m + n) ** 2 * (
                    old_mean - new_mean) ** 2)
            self.n_samples += n
        return


def make_optimizer(parameters, fedavg_optim=False):
    if not fedavg_optim:
        if cfg['optim_name'] == 'SGD':
            optimizer = optim.SGD(parameters, lr=cfg['lr'], momentum=cfg['momentum'],
                                weight_decay=cfg['weight_decay'], nesterov=cfg['nesterov'])
        elif cfg['optim_name'] == 'Adam':
            optimizer = optim.Adam(parameters, lr=cfg['lr'], betas=cfg['betas'],
                                weight_decay=cfg['weight_decay'])
        elif cfg['optim_name'] == 'LBFGS':
            optimizer = optim.LBFGS(parameters, lr=cfg['lr'])
        else:
            raise ValueError('Not valid optimizer name')
    else:
        lr = 1
        momentum = 0.5
        betas = (0.9, 0.999)
        weight_decay = 0
        nesterov = False
        if cfg['optim_name'] == 'SGD':
            optimizer = optim.SGD(parameters, lr=lr, momentum=momentum,
                                weight_decay=weight_decay, nesterov=nesterov)
        elif cfg['optim_name'] == 'Adam':
            optimizer = optim.Adam(parameters, lr=cfg['lr'], betas=betas,
                                weight_decay=weight_decay)
        elif cfg['optim_name'] == 'LBFGS':
            optimizer = optim.LBFGS(parameters, lr=lr)
        else:
            raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer):
    if cfg['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif cfg['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['factor'])
    elif cfg['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'],
                                                   gamma=cfg['factor'])
    elif cfg['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_rounds'], eta_min=0)
    elif cfg['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg['factor'],
                                                         patience=cfg['patience'], verbose=False,
                                                         threshold=cfg['threshold'], threshold_mode='rel',
                                                         min_lr=cfg['min_lr'])
    elif cfg['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg['lr'], max_lr=10 * cfg['lr'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


def resume(model_tag, model_dir, load_tag='checkpoint', verbose=True):
    if os.path.exists('{}/{}_{}.pt'.format(model_dir, model_tag, load_tag)):
        result = load('{}/{}_{}.pt'.format(model_dir, model_tag, load_tag))
    else:
        FileExistsError('Not exists model tag: {}'.format(model_tag))
    if verbose:
        if load_tag == 'checkpoint':
            print('Resume from epoch {}'.format(result['epoch']))
        else:
            print('Loaded from epoch {}'.format(result['epoch'] - 1))
    return result


def collate(input):
    for k in input:
        input[k] = torch.stack(input[k], 0)
    return input
