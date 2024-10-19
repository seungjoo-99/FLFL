import datetime
import models
import os
import shutil
import time
import torch
from torch.utils.data import Subset
import torch.backends.cudnn as cudnn
import numpy as np
from config import cfg, process_args
from data import get_dataset, make_data_loader, split_dataset, make_sbn_stats
from metrics import Metric
from modules import Server, Client
from utils import save, to_device, process_control, make_optimizer, make_scheduler, resume, collate, metric_batch, eval_only, get_model, check_algorithm, select_clients
from logger import make_logger
import wandb
from modules.utils import save_model_state_dict
from algorithm import get_algorithm
import random
from copy import deepcopy


cudnn.benchmark = False
process_args()

model_path = './output/flfl/model'

def main():
    process_control()
    seeds = list(range(cfg['seed'], cfg['seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data']['name'], cfg['model']['name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        if cfg['eval_only']:
            eval_only(model_path)
            continue
        runExperiment()
    return

def set_seed():
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(cfg['seed'])


def runExperiment():
    set_seed()
    
    dataset = get_dataset(include_lb_to_ulb=False)

    model = get_model(track=True)

    algorithm = get_algorithm()
    data_loader = make_data_loader(dataset, only_eval=True)
    optimizer = make_optimizer(model.parameters())
    scheduler = make_scheduler(optimizer)

    if check_algorithm('flfl'):
        metric = Metric({'train_clients': ['Loss', 'Accuracy', 'PAccuracy', 'MAccuracy', 'LabelRatio', 'SMAccuracy', 'SLabelRatio'],
                            'train_server': ['Loss', 'Accuracy'],
                            'test': ['Loss', 'Accuracy']})
    else:
        metric = Metric({'train_clients': ['Loss', 'Accuracy', 'PAccuracy', 'MAccuracy', 'LabelRatio'],
                            'train_server': ['Loss', 'Accuracy'],
                            'test': ['Loss', 'Accuracy']})


    server_dataset = {'train_lb': dataset['train_lb'], 'train_ulb': []}
    
    last_epoch = 1
    client_datasets, client_datasets_idx = split_dataset(dataset)
    server = make_server(model)
    clients = make_clients(model)
    logger = make_logger()

    for sp in range(5):
        print(f'Client {sp}: {client_datasets_idx[sp]["train_ulb"][:5]}')

    set_seed()
    selected_clients = select_clients(cfg['control_fl']['num_clients'], cfg['control_fl']['active_rate'], cfg['num_rounds'])
    assert len(selected_clients) == cfg['num_rounds']

    for sp in range(5):
        print(f'Selected Clients {sp}: {selected_clients[sp][:5]}')
    
    if cfg['server_pretrain'] == 1:
        if check_algorithm('flfl')  and cfg['data']['name'] == 'svhn':
            pretrain_iters = 2048
            num_iter_per_epoch = len(dataset['train_lb']) // cfg['batch_size']
            num_pretrain_epochs = pretrain_iters // (num_iter_per_epoch * cfg['num_epochs'])
        elif check_algorithm('flfl') and cfg['data']['name'] == 'cifar10':
            num_pretrain_epochs = 10

    print(f'Server Pretrain: {num_pretrain_epochs}')
    for i in range(int(num_pretrain_epochs)):
        train_server(server_dataset, server, optimizer, metric)

    model.load_state_dict(server.model_state_dict, strict=False)
    model = make_sbn_stats(dataset, model)
    server.model_state_dict = save_model_state_dict(model.state_dict())
    for epoch in range(last_epoch, cfg['num_rounds'] + 1):
        logger.reset()
        print('Round {}, Model: {}'.format(epoch, cfg['model_tag']))
        server.distribute(clients, selected_clients[epoch - 1])
        train_clients(client_datasets, clients, optimizer, metric, logger, epoch)
        if cfg['setting'] == 'lb-at-clients':
            server.update(clients)
        elif cfg['setting'] == 'lb-at-server':
            if cfg['control_fl']['global_ft'] == 0:
                train_server(server_dataset, server, optimizer, metric, logger, epoch)
                server.update_parallel(clients)
            else:
                server.update(server_dataset, clients)
                train_server(server_dataset, server, optimizer, metric, logger, epoch)
        
        model.load_state_dict(server.model_state_dict, strict=False)
        if cfg['control_fl']['sbn'] > 0:
            model = make_sbn_stats(dataset, model)
        server.model_state_dict = save_model_state_dict(model.state_dict())
        test(data_loader, model, algorithm, metric, logger, epoch)
        logger.save()

    logger.close()
    return


def make_server(model):
    server = Server(model)
    return server


def make_clients(model, unsup_model=None):
    clients = []
    for client_id in range(cfg['control_fl']['num_clients']):
        client = Client(client_id, model, unsup_model)
        clients.append(client)
    return clients


def train_clients(client_datasets, clients, optimizer, metric, logger, epoch):
    lr = optimizer.param_groups[0]['lr']
    active_client_ids = [id for id, client in enumerate(clients) if client.active]
    num_active_clients = len(active_client_ids)
    for i, active_client_id in enumerate(active_client_ids):
        clients[active_client_id].train(client_datasets[active_client_id], lr, metric, logger, epoch)
        exp_progress = 100. * (i + 1) / num_active_clients
        info = {'info': ['Training round {} (C) ({:.0f}%)'.format(epoch, exp_progress),
                        # 'Learning rate: {:.6f}'.format(lr),
                        'ID: {}'.format(active_client_id)]}
        logger.print_info('train_clients', metric.metric_name['train_clients'], info)
    logger.write('train_clients', metric.metric_name['train_clients'], epoch)
    return


def train_server(dataset, server, optimizer, metric, logger=None, epoch=0):
    lr = optimizer.param_groups[0]['lr']
    server.train(dataset, lr, metric, logger)
    info = {'info': ['Training round {} (S)'.format(epoch)]}
                            # 'Learning rate: {:.6f}'.format(lr),
                            # 'ID: {}({}/{})'.format(active_client_id, i + 1, num_active_clients)]}
    if logger is not None:
        logger.print_info('train_server', metric.metric_name['train_server'], info)
        logger.write('train_server', metric.metric_name['train_server'], epoch)
    return


def test(data_loader, model, algorithm, metric, logger, epoch):
    with torch.no_grad():
        model.eval()
        for data_lb in data_loader['eval']:
            data_lb = to_device(data_lb, cfg['device'])
            kwargs = {'test': True}
            out_dict = algorithm.forward(model, data_lb=data_lb, **kwargs)
            in_dict = metric_batch(data_lb=data_lb)
            evaluation = metric.evaluate(metric.metric_name['test'], in_dict, out_dict)
            logger.append(evaluation, 'test', n=in_dict['targets'].shape[0])
        info = {'info': ['TESTING', 'Model: {}'.format(cfg['model_tag']), 'Test round: {}'.format(epoch)]}
        logger.print_info('test', metric.metric_name['test'], info)
        logger.write('test', metric.metric_name['test'], epoch)
    return


def save_checkpoint_and_best(metric, **result):
    logger = result['logger']
    save(result, '{}/{}_checkpoint.pt'.format(model_path, cfg['model_tag']))
    if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
        metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
        shutil.copy('{}/{}_checkpoint.pt'.format(model_path, cfg['model_tag']),
                    '{}/{}_best.pt'.format(model_path, cfg['model_tag']))

if __name__ == "__main__":
    main()
