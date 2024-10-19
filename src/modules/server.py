import copy
import models
import numpy as np
import torch
from config import cfg
from data import make_data_loader
from utils import make_optimizer, collate, to_device, metric_batch, get_model, zip_data_loader, check_algorithm
from .utils import save_model_state_dict, save_optimizer_state_dict
from algorithm import *
from torch.utils.data import Subset
from algorithm.sam import get_sam_optimizer
from copy import deepcopy

class Server:
    def __init__(self, model):
        self.model_state_dict = save_model_state_dict(model.state_dict())
        optimizer = make_optimizer(model.parameters())
        fedavg_optimizer = make_optimizer(model.parameters(), fedavg_optim=True)
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.fedavg_optimizer_state_dict = save_optimizer_state_dict(fedavg_optimizer.state_dict())
        self.algorithm = get_algorithm()


    def distribute(self, clients, client_id):
        num_active_clients = len(client_id)
        for i in range(num_active_clients):
            client = clients[client_id[i]]
            client.active = True
            client.model_state_dict = copy.deepcopy(self.model_state_dict)
        return
    
    @torch.no_grad()
    def fedavg(self, client):
        if len(client) > 0:
            weight = torch.ones(len(client))
            weight = weight / weight.sum()
            for k, v in self.model_state_dict.items():
                parameter_type = k.split('.')[-1]
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    tmp_v = v.data.new_zeros(v.size())
                    for m in range(len(client)):
                        tmp_v += weight[m] * client[m].model_state_dict[k]
                    self.model_state_dict[k] = tmp_v

    
    def semifl_avg(self, server_dataset, valid_client, parallel):
        model = get_model(to_device=False)
        model.load_state_dict(self.model_state_dict, strict=False)
        fedavg_optimizer = make_optimizer(model.parameters(), fedavg_optim=True)
        fedavg_optimizer.load_state_dict(self.fedavg_optimizer_state_dict)
        fedavg_optimizer.zero_grad()
        

        if parallel:
            weight = torch.ones(len(valid_client))
            weight = weight / (2 * (weight.sum() - 1))
            weight[0] = 1 / 2 if len(valid_client) > 1 else 1
        else:
            if check_algorithm('flfl'):
                weight = torch.ones(len(valid_client))
                for m in range(len(valid_client)):
                    weight[m] = 1 - valid_client[m].algorithm.global_t
                weight = weight / weight.sum()
            else:
                weight = torch.ones(len(valid_client))
                weight = weight / weight.sum()

        tmp_vs = {}

        with torch.no_grad():
            for k, v in model.named_parameters():
                parameter_type = k.split('.')[-1]
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    tmp_v = v.data.new_zeros(v.size())
                    for m in range(len(valid_client)):
                        tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                    tmp_vs[k] = tmp_v

                
        with torch.no_grad():
            for k, v in model.named_parameters():
                if k in tmp_vs:
                    v.grad = (v.data - tmp_vs[k]).detach()
        fedavg_optimizer.step()
        self.fedavg_optimizer_state_dict = save_optimizer_state_dict(fedavg_optimizer.state_dict())
        self.model_state_dict = save_model_state_dict(model.state_dict())

    def update(self, server_dataset, client):
       
        valid_client = [client[i] for i in range(len(client)) if client[i].active]
        if len(valid_client) > 0:
            # vanilla model aggregation from FedAvg, which is diferent from the orignal
            if cfg['use_fedavg'] == 1:
                self.fedavg(valid_client)
            else:
                self.semifl_avg(server_dataset, valid_client, parallel=False)
        
        for i in range(len(client)):
            client[i].active = False

        return

    def update_parallel(self, client):
        # if 'frgd' not in cfg['loss_mode']:
        with torch.no_grad():
            valid_client_server = [self] + [client[i] for i in range(len(client)) if client[i].active]
            if len(valid_client_server) > 0:
                # vanilla model aggregation from FedAvg, which is diferent from the orignal 
                if cfg['control_fl']['use_fedavg'] == 1:
                    self.fedavg(valid_client_server)
                else:
                    self.semifl_avg(valid_client_server, parallel=True)
        for i in range(len(client)):
            client[i].active = False
        return

    def train(self, dataset, lr, metric, logger):
        model = get_model()
        # model = eval('models.{}().to(cfg["device"])'.format(cfg['model']['type']))
        model.load_state_dict(self.model_state_dict, strict=False)
        data_loader = make_data_loader(dataset)
        self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer = make_optimizer(model.parameters())
        optimizer.load_state_dict(self.optimizer_state_dict)
        kwargs = {}
        
        model.train()


        for epoch in range(1, cfg['num_epochs'] + 1):
            for i, data_batch in enumerate(zip_data_loader(data_loader)):
                data_batch = to_device(data_batch, cfg['device'])
                out_dict = self.algorithm.forward(model, data_batch=data_batch, **kwargs)
                optimizer.zero_grad()
                out_dict['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                optimizer.step()
                
                in_dict = metric_batch(data_batch)
                evaluation = metric.evaluate(metric.metric_name['train_server'], in_dict, out_dict)
                if logger is not None:
                    logger.append(evaluation, 'train_server', n=in_dict['targets'].shape[0])
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.model_state_dict = save_model_state_dict(model.state_dict())
        return

