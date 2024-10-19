from config import cfg
import torch
import torch.nn.functional as F

def kldiv_with_mask(logits_s, logits_w, mask, temp):
    logits_s = F.log_softmax(logits_s / temp, dim=1)
    logits_w = F.softmax(logits_w.detach(), dim=1)

    if mask is None:
        kl_loss = F.kl_div(logits_s, logits_w, reduction='batchmean')
    else:
        kl_loss = F.kl_div(logits_s, logits_w, reduction='none')
        kl_loss = kl_loss * mask.unsqueeze(dim=-1).repeat(1, logits_s.shape[1])
        kl_loss = kl_loss.sum(dim=1).mean()

    return kl_loss


def FixMatchLoss(input, target=None):
    logits_x_ulb_w, logits_x_ulb_s = input['logits'].chunk(2)
    pseudo_label, mask = FixmatchPseudoLabel(logits_x_ulb_w)
    unsup_loss = consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask)
    return unsup_loss

@torch.no_grad()
def FixmatchPseudoLabel(logits_x_ulb_w):
    max_probs_w, max_idx_w = torch.max(torch.softmax(logits_x_ulb_w, dim=-1), dim=-1)
    mask = max_probs_w.ge(0.95)

    return max_idx_w.detach(), mask

def ce_loss(logits, targets, reduction='mean'):
    # check logits is dict or not
    if isinstance(logits, dict):
        logits = logits['logits']

    return F.cross_entropy(logits, targets, reduction=reduction)
    
def consistency_loss(logits, targets, name='ce', mask=None):
    if mask is not None:
        logits = logits[mask]
        targets = targets[mask]
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    else:
        loss = F.cross_entropy(logits, targets, reduction='none')

    if  mask is not None and cfg['maskout_ulb']:
        if any(mask):
            loss = loss.mean()
        else:
            loss = loss.sum()
    else:
        loss = loss.sum() / targets.shape[0]
    return loss
