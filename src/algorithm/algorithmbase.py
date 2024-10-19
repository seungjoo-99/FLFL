from config import cfg
import torch
import torch.nn.functional as F
from inspect import signature
from .sam import SAM
from .criterion import ce_loss, consistency_loss


def get_fullysupervised():
    return AlgorithmBase()

class AlgorithmBase:
    def __init__(self):
        self.use_cat = cfg['use_cat']
        self.num_classes = cfg['data']['num_classes']
        self.lambda_u = cfg['ulb_loss_ratio']

        self.ce_loss = ce_loss
        self.consistency_loss = consistency_loss
        
        if cfg['control_ssl']['mixup'] > 0:
            alpha = 0.75
            self.beta = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
    
    def forward(self, model, data_batch=None, data_lb=None, data_ulb=None, data_ulb_mix=None, **kwargs):
        if data_batch is not None:
            data_lb, data_ulb = data_batch[:2]
            data_ulb_mix = None if len(data_batch) < 3 else data_batch[2]
        if 'test' in kwargs:
            out_dict = self.forward_lb_base(model, data_lb)
        elif data_lb is not None and data_ulb is not None:
            out_dict =  self.forward_lb_ulb(model, data_lb, data_ulb, **kwargs)
        elif data_ulb is not None:
            out_dict = self.forward_ulb(model, data_ulb, **kwargs)
        elif data_lb is not None:
            out_dict = self.forward_lb(model, data_lb, **kwargs)
        else:
            raise ValueError('No data to train')
        if data_ulb_mix != None:
            out_dict = self.forward_ulb_mix(model, data_ulb, data_ulb_mix, out_dict)
        return out_dict
    
    def forward_lb_base(self, model, data_lb):
        x_lb = data_lb['x_lb']
        y_lb = data_lb['y_lb']
        
        outs_x_lb = model(x_lb) 
        logits_x_lb = outs_x_lb['logits']

        
        sup_loss = self.ce_loss(logits_x_lb, y_lb)
        out_dict = self.process_out_dict(loss=sup_loss, logits=logits_x_lb, targets=y_lb)
        return out_dict
    
    def forward_lb(self, model, data_lb, **kwargs):
        return self.forward_lb_base(model, data_lb)
    
    def forward_lb_ulb(self, model, data_lb, data_ulb):
        raise NotImplementedError
    
    def forward_ulb(self, model, data_ulb, **kwargs):
        raise NotImplementedError

    def forward_ulb_mix(self, model, data_ulb, data_ulb_mix, out_dict):
        lam_mix = self.beta.sample()[0]
        x_ulb_mix_w = lam_mix * data_ulb['x_ulb_w'] + (1-lam_mix) * data_ulb_mix['x_ulb_w']
        outs_x_ulb_mix_w = model(x_ulb_mix_w)
        logits_x_ulb_mix_w = outs_x_ulb_mix_w['logits']
        pseudo_label = data_ulb['pseudo_y_ulb']
        pseudo_label_mix = data_ulb_mix['pseudo_y_ulb']
        mix_loss = lam_mix * self.consistency_loss(logits_x_ulb_mix_w, pseudo_label) + (1 - lam_mix) * self.consistency_loss(logits_x_ulb_mix_w, pseudo_label_mix)
        out_dict['loss'] += mix_loss
        return out_dict

    def process_out_dict(self, out_dict=None, **kwargs):
        """
        process the out_dict as return of forward
        """
        if out_dict is None:
            out_dict = {}

        for arg, var in kwargs.items():
            out_dict[arg] = var

        # process res_dict, add output from res_dict to out_dict if necessary
        return out_dict

    def process_log_dict(self, log_dict=None, prefix="train", **kwargs):
        """
        process the tb_dict as return of forward
        """
        if log_dict is None:
            log_dict = {}

        for arg, var in kwargs.items():
            log_dict[f"{prefix}/" + arg] = var
        return log_dict
