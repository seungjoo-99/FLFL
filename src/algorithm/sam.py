import torch
from torch.nn.modules.batchnorm import _BatchNorm
from config import cfg
from collections import defaultdict

def get_sam_optimizer(model, rho=None):
    if rho is None:
        rho = cfg['control_sam']['rho']

    if cfg['control_sam']['opt'] == 'sam':
        return SAM(model, torch.optim.SGD, rho=rho)
    elif cfg['control_sam']['opt'] == 'asam':
        return ASAM(model, torch.optim.SGD, rho=rho)
    else:
        raise ValueError('Not valid SAM optimizer')

class ASAM:
    def __init__(self, model, base_optimizer, rho=0.5, eta=0.01, **kwargs):

        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)

        if zero_grad:
            self.model.zero_grad()

    @torch.no_grad()
    def second_step(self):
        for n, p in self.model.named_parameters():
            if "eps" not in self.state[p]: #p.grad is None or p not in self.state:
                continue
            p.sub_(self.state[p]["eps"])


class SAM(torch.optim.Optimizer):
    def __init__(self, model, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        kwargs['lr'] = 0.001 #dummy value
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        params = model.parameters()
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                # if p.grad is None: continue
                if "old_p" not in self.state[p]: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
