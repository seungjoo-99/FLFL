import torch
import torch.nn.functional as F
from config import cfg
from data import make_data_loader
from .utils import gen_ulb_targets, replace_inf_to_zero
from .algorithmbase import AlgorithmBase
from utils import to_device, get_model
import models
import numpy as np


def get_fixmatch():
    return FixMatch()

class FixMatch(AlgorithmBase):
    def __init__(self, ):
        super().__init__() 
        self.T = 0.95
    def forward_ulb(self, model, data_ulb, **kwargs):
        x_ulb_w = data_ulb['x_ulb_w']
        x_ulb_s = data_ulb['x_ulb_s']
        if cfg['control_fl']['global_pseudo_lb'] > 0:
            outs_x_ulb_s = model(x_ulb_s)
            logits_x_ulb_s = outs_x_ulb_s['logits']
            mask = data_ulb['pseudo_y_ulb'] != -1
            pseudo_label = data_ulb['pseudo_y_ulb']
            unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask)
        else: # curently not tailored yet
            pass
        total_loss = self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, logits=logits_x_ulb_s,
                                         mask=mask, pseudo_lb=pseudo_label)
        return out_dict

    @torch.no_grad()
    def make_pseudo_lb_dataset(self, dataset, model_state_dict, global_epoch=None, return_probs=False):
        data_loader = make_data_loader(dataset, shuffle=False, train=False)
        model = get_model(track=True)
        model.load_state_dict(model_state_dict, strict=False)
        model.eval()
        logits_x_ulb = []
        for data_ulb in data_loader["train_ulb"]:
            data_ulb = to_device(data_ulb, cfg['device'])
            x_ulb = data_ulb['x_ulb_w']
            outs_x_ulb = model(x_ulb)
            logits_x_ulb.append(outs_x_ulb['logits'].cpu())
        
        logits_x_ulb = torch.cat(logits_x_ulb, dim=0)
        probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        mask = max_probs.ge(self.T)
        pseudo_targets = max_idx.masked_fill(mask == False, -1)
        pseudo_lb_idx_in_subset = np.where(mask)[0].tolist()
        pseudo_lb_idx = [dataset['train_ulb'].indices[i] for i in pseudo_lb_idx_in_subset]
        dataset['train_ulb'].dataset.pseudo_targets[dataset['train_ulb'].indices] = pseudo_targets

        if return_probs:
            return pseudo_lb_idx, max_probs[mask]
        return pseudo_lb_idx
