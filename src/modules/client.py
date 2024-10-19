import copy
import models
import numpy as np
import torch
from torch.utils.data import Subset
import torch.nn.functional as F
from itertools import compress
from config import cfg
from data import make_data_loader
from utils import make_optimizer, collate, to_device, metric_batch, get_model, zip_data_loader, check_algorithm
from .utils import save_model_state_dict, save_optimizer_state_dict
from algorithm import *
import wandb


class Client:
    def __init__(self, client_id, model, unsup_model=None):
        self.client_id = client_id
        self.model_state_dict = save_model_state_dict(model.state_dict())
        self.algorithm = get_algorithm()
        optimizer = make_optimizer(model.parameters())
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.active = False
        self.beta = torch.distributions.beta.Beta(torch.tensor([cfg['alpha']]), torch.tensor([cfg['alpha']]))
        self.verbose = cfg['verbose']

    def train(self, dataset, lr, metric, logger, global_epoch):
        model = get_model()
        model.load_state_dict(self.model_state_dict, strict=False)
        pseudo_lb_idx = None
        pseudo_dataset = dataset


        if cfg['control_fl']['global_pseudo_lb'] > 0:
            # global_pseudo_labeling but keep non-pseudo-labeled samples in batches
            pseudo_lb_idx = self.algorithm.make_pseudo_lb_dataset(dataset, self.model_state_dict, global_epoch)
            if cfg['control_fl']['global_pseudo_lb'] == 2: #global_pseudo_labeling and remove non-pseudo-labeled samples from batches
                pseudo_dataset = {'train_lb': dataset['train_lb'], 'train_ulb': Subset(dataset['train_ulb'].dataset, pseudo_lb_idx)}
                if cfg['control_ssl']['mixup'] > 0:
                    mix_pseudo_lb_idx = np.random.choice(pseudo_lb_idx, size=len(pseudo_lb_idx), replace=cfg['control_ssl']['mixup'])
                    pseudo_dataset['train_ulb_mix'] = Subset(dataset['train_ulb'].dataset, mix_pseudo_lb_idx)
        data_loader = make_data_loader(pseudo_dataset)
        if data_loader['train_lb'] is None and data_loader['train_ulb'] is None:
            return
        self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer = make_optimizer(model.parameters())
        optimizer.load_state_dict(self.optimizer_state_dict)
        model.train()
        kwargs = {'epoch': global_epoch}
       

        for epoch in range(1, cfg['num_epochs'] + 1):
            for i, data_batch in enumerate(zip_data_loader(data_loader)):
                data_batch = to_device(data_batch, cfg['device'])
                out_dict = self.algorithm.forward(model, data_batch=data_batch, **kwargs)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                in_dict = metric_batch(data_batch)
                evaluation = metric.evaluate(metric.metric_name['train_clients'], in_dict, out_dict)
                
                if cfg['control_fl']['global_pseudo_lb'] == 2:
                    evaluation['LabelRatio'] = len(pseudo_lb_idx) / len(dataset['train_ulb'])
                    evaluation['PAccuracy'] *= evaluation['LabelRatio']
                logger.append(evaluation, 'train_clients', n=in_dict['targets'].shape[0])

        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.model_state_dict = save_model_state_dict(model.state_dict())
        return
