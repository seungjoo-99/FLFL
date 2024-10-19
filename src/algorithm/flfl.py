import torch
import torch.nn.functional as F
from config import cfg
from data import make_data_loader
from .utils import gen_ulb_targets, replace_inf_to_zero
from .algorithmbase import AlgorithmBase
from utils import to_device, get_model
import models
from algorithm import *
from .sam import get_sam_optimizer
from .criterion import kldiv_with_mask
from utils import check_algorithm
import numpy as np



def get_flfl():
    return FLFL()

class FLFL(AlgorithmBase):
    def __init__(self, ):
        super().__init__()
        self.flfl_T = cfg['flfl_T']
        self.kldiv_with_mask = kldiv_with_mask

        self.T = cfg['T']
        self.use_hard_label = cfg['hard_label']
        self.ema_p = cfg['ema_p']
        self.use_quantile = cfg['use_quantile']
        self.clip_thresh = cfg['clip_thresh']
        
        self.m = 0.999
        self.local_t = None
        self.label_hist = None
        self.global_t = None

    def forward_ulb(self, model, data_ulb, **kwargs):
        sam_optimizer = get_sam_optimizer(model)
        model.zero_grad()
        x_ulb_s = data_ulb['x_ulb_s']
        x_ulb_w = data_ulb['x_ulb_w']
        mask = data_ulb['pseudo_y_ulb'] != -1
        fix_mask = data_ulb['fix_pseudo_y_ulb'] != -1
        pseudo = data_ulb['pseudo_y_ulb']
    
        with torch.no_grad():
            logits_x_ulb_w = model(x_ulb_w)['logits'].detach()
            logits_x_ulb_s = model(x_ulb_s)['logits'].detach()
            
            logits_weak = logits_x_ulb_w if cfg['use_weak'] else logits_x_ulb_s

        data_ulb['pseudo_y_ulb'] = torch.where(fix_mask, pseudo, -1)
        out_dict = self.forward_ulb_cat(model, data_ulb)
        out_dict['loss'].backward()

        sam_optimizer.first_step(zero_grad=True)

        logits_ulb_s_hat = model(x_ulb_s)['logits']
        kl_loss = self.kldiv_with_mask(logits_ulb_s_hat, logits_weak, fix_mask, self.flfl_T)
        kl_grad = torch.autograd.grad(kl_loss, model.parameters())

        sam_optimizer.second_step()
        model.zero_grad()

        data_ulb['pseudo_y_ulb'] = pseudo
        out_dict_2 = self.forward_ulb_cat(model, data_ulb)

        out_dict_2['loss'].backward()

        for param, grad in zip(model.parameters(), kl_grad):
            param.grad += grad

        out_dict_2['strong_mask'] = fix_mask

        return out_dict_2

    def forward_ulb_cat(self, model, data_ulb, **kwargs):
        x_ulb_w = data_ulb['x_ulb_w']
        x_ulb_s = data_ulb['x_ulb_s']
        if cfg['control_fl']['global_pseudo_lb'] > 0:
            outs_x_ulb_s = model(x_ulb_s)
            logits_x_ulb_s = outs_x_ulb_s['logits']
            mask = data_ulb['pseudo_y_ulb'] != -1
            pseudo_label = data_ulb['pseudo_y_ulb']
            unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask)

        else: # curently not tailored yet
            # inference and calculate sup/unsup losses
            if self.use_cat:
                inputs = torch.cat((x_ulb_w, x_ulb_s))
                outputs = model(inputs)
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'].chunk(2)

            else:
                outs_x_ulb_s = model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']

                with torch.no_grad():
                    outs_x_ulb_w = model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                
            # calculate mask
            mask = self.sat_masking(logits_x_ulb=logits_x_ulb_w)


            # generate unlabeled targets using pseudo label hook
            pseudo_label = gen_ulb_targets(logits=logits_x_ulb_w, use_hard_label=self.use_hard_label, T=self.T)
            
            # calculate unlabeled loss
            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                            pseudo_label,
                                            'ce',
                                            mask=mask)

        total_loss = self.lambda_u * unsup_loss

        # out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
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

        self.global_t = max_probs.mean()
        self.local_t = probs_x_ulb.mean(dim=0) 
        hist = torch.bincount(max_idx.reshape(-1), minlength=self.local_t.shape[0]).to(self.local_t.dtype) 
        self.label_hist = hist / hist.sum()
        
        mod = self.local_t / torch.max(self.local_t, dim=-1)[0]
        mask = max_probs.ge(self.global_t * mod[max_idx])


        fix_mask = max_probs.ge(0.95)
        fix_pseudo_targets = max_idx.masked_fill(fix_mask == False, -1)
        fix_pseudo_lb_idx_in_subset = np.where(fix_mask)[0].tolist()
        fix_pseudo_lb_idx = [dataset['train_ulb'].indices[i] for i in fix_pseudo_lb_idx_in_subset]
        dataset['train_ulb'].dataset.fix_pseudo_targets[dataset['train_ulb'].indices] = fix_pseudo_targets

        pseudo_targets = max_idx.masked_fill(mask == False, -1)
        pseudo_lb_idx_in_subset = np.where(mask)[0].tolist()
        pseudo_lb_idx = [dataset['train_ulb'].indices[i] for i in pseudo_lb_idx_in_subset]
        dataset['train_ulb'].dataset.pseudo_targets[dataset['train_ulb'].indices] = pseudo_targets

        if return_probs:
            return pseudo_lb_idx, max_probs[mask]

        return pseudo_lb_idx

    @torch.no_grad()
    def sat_masking(self, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):
        if not self.local_t.is_cuda:
            self.local_t = self.local_t.to(logits_x_ulb.device)
        if not self.label_hist.is_cuda:
            self.label_hist = self.label_hist.to(logits_x_ulb.device)
        if not self.global_t.is_cuda:
            self.global_t = self.global_t.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        self.sat_update(probs_x_ulb)

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        mod = self.local_t / torch.max(self.local_t, dim=-1)[0]
        mask = max_probs.ge(self.global_t * mod[max_idx]).to(max_probs.dtype)
        return mask.to(torch.int)
    
    @torch.no_grad()
    def sat_update(self, probs_x_ulb):
        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1,keepdim=True)

        if self.use_quantile:
            self.global_t = self.global_t * self.m + (1 - self.m) * torch.quantile(max_probs,0.8) #* max_probs.mean()
        else:
            self.global_t = self.global_t * self.m + (1 - self.m) * max_probs.mean()
        
        if self.clip_thresh:
            self.global_t = torch.clip(self.global_t, 0.0, 0.95)

        self.local_t = self.local_t * self.m + (1 - self.m) * probs_x_ulb.mean(dim=0)
        hist = torch.bincount(max_idx.reshape(-1), minlength=self.local_t.shape[0]).to(self.local_t.dtype) 
        self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum())
