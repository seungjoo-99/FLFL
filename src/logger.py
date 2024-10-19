from collections import defaultdict
from collections.abc import Iterable
from numbers import Number
from utils import ntuple, cfg
import wandb


class Logger:
    def __init__(self):
        self.tracker = defaultdict(int)
        self.counter = defaultdict(int)
        self.mean = defaultdict(int)
        self.history = defaultdict(list)

        tags = [f"batch:{cfg['ulb_batch_size']}", 
                f"num_label:{cfg['control_ssl']['num_labels']}", 
                f"data_split:{cfg['control_fl']['data_split_mode']}",
                f"dataset:{cfg['data']['name']}"]
        
        run_name = f"seed{cfg['seed']}_"
        run_name += f"batch{cfg['ulb_batch_size']}_"
        run_name += cfg['control_ssl']['algorithm']
        if cfg['control_sam']['opt']:
            run_name += '_' + '_'.join([str(cfg['control_sam'][k]) for k in cfg['control_sam'] if cfg['control_sam'][k]])
        
        # Init wandb
        wandb.init(entity=cfg['log']['entity'],
                   project=cfg['log']['project'],
                   mode=cfg['log']['mode'],
                   config=cfg,
                   name=run_name, #cfg['log']['name'],
                   notes=cfg['log']['note'],
                   tags=tags,)

    def save(self):
        for name in self.mean:
            self.history[name].append(self.mean[name])
        return

    def reset(self):
        self.tracker = defaultdict(int)
        self.counter = defaultdict(int)
        self.mean = defaultdict(int)
        return

    def append(self, result, tag, n=1, mean=True):
        for k in result:
            name = '{}/{}'.format(tag, k)
            self.tracker[name] = result[k]
            if mean:
                if isinstance(result[k], Number):
                    self.counter[name] += n
                    self.mean[name] = ((self.counter[name] - n) * self.mean[name] + n * result[k]) / self.counter[name]
                elif isinstance(result[k], Iterable):
                    if name not in self.mean:
                        self.counter[name] = [0 for _ in range(len(result[k]))]
                        self.mean[name] = [0 for _ in range(len(result[k]))]
                    _ntuple = ntuple(len(result[k]))
                    n = _ntuple(n)
                    for i in range(len(result[k])):
                        self.counter[name][i] += n[i]
                        self.mean[name][i] = ((self.counter[name][i] - n[i]) * self.mean[name][i] + n[i] *
                                              result[k][i]) / self.counter[name][i]
                else:
                    raise ValueError('Not valid data type')
        return

    def write(self, tag, metric_names, epoch):
        names = ['{}/{}'.format(tag, k) for k in metric_names]
        evaluation_info = []
        for name in names:
            tag, k = name.split('/')
            if isinstance(self.mean[name], Number):
                s = self.mean[name]
                evaluation_info.append('{}: {:.4f}'.format(k, s))
                wandb.log({name: s}, step=epoch)
            elif isinstance(self.mean[name], Iterable):
                s = tuple(self.mean[name])
                evaluation_info.append('{}: {}'.format(k, s))
                wandb.log({name: s[0]}, step=epoch)
            else:
                raise ValueError('Not valid data type')
        return
    
    def print_info(self, tag, metric_names, info):
        print(' | '.join(info['info']))
        metric_log_list = ['{}: {:.3f}'.format(metric_name, self.mean[f'{tag}/{metric_name}']) for metric_name in metric_names]
        metric_log = ', '.join(metric_log_list)
        print(f'\t{metric_log}')
        
    def close(self):
        wandb.finish()


def make_logger():
    logger = Logger()
    return logger
