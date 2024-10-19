import yaml
import argparse

global cfg
if 'cfg' not in globals():
    with open('config/config.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

def process_args():
    parser = argparse.ArgumentParser(description='cfg')
    for k in cfg:
        exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
    parser.add_argument('--control_name', default=None, type=str)
    args = vars(parser.parse_args())

    for k in cfg:
        cfg[k] = args[k]
    if cfg['control_ssl']['algorithm'] == 'fullysupervised':
        cfg['control_name'] = cfg['control_ssl']['algorithm']
        cfg['control_name'] += '_{}'.format('all' if cfg['control_ssl']['num_labels'] < 0 else cfg['control_ssl']['num_labels'])
    else:
        cfg['control_name'] = '_'.join([str(cfg['control_ssl'][k]) for k in cfg['control_ssl'] if cfg['control_ssl'][k]])
        if cfg['control_sam']['opt']:
            cfg['control_name'] += '_' + '_'.join([str(cfg['control_sam'][k]) for k in cfg['control_sam'] if cfg['control_sam'][k]])
    if cfg['setting'] != 'centralized':
        cfg['control_name'] += '_{}_'.format(cfg['setting']) + '_'.join([str(cfg['control_fl'][k]) for k in cfg['control_fl']])

        
    return
