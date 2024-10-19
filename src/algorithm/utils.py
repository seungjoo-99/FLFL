import torch
from torch.nn.modules.batchnorm import _BatchNorm
from config import cfg

def compute_probs(logits):
    return torch.softmax(logits, dim=-1)

def gen_ulb_targets(logits, 
                    use_hard_label=True, 
                    T=1.0,
                    softmax=True, # whether to compute softmax for logits, input must be logits
                    label_smoothing=0.0):
    
    """
    generate pseudo-labels from logits/probs

    Args:
        logits: logits (or probs, need to set softmax to False)
        use_hard_label: flag of using hard labels instead of soft labels
        T: temperature parameters
        softmax: flag of using softmax on logits
        label_smoothing: label_smoothing parameter
    """
    with torch.no_grad():
        logits = logits.detach()
        if use_hard_label:
            # return hard label directly
            pseudo_label = torch.argmax(logits, dim=-1)
            if label_smoothing:
                pseudo_label = smooth_targets(logits, pseudo_label, label_smoothing)
            return pseudo_label
        
        # return soft label
        if softmax:
            # pseudo_label = torch.softmax(logits / T, dim=-1)
            pseudo_label = compute_probs(logits / T)
        else:
            # inputs logits converted to probabilities already
            pseudo_label = logits
        return pseudo_label

def smooth_targets(logits, targets, smoothing=0.1):
    """
    label smoothing
    """
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / (logits.shape[-1] - 1))
        true_dist.scatter_(1, targets.data.unsqueeze(1), (1 - smoothing))
    return true_dist

def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val
