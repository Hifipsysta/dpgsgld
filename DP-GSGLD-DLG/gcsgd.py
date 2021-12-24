

import math
import torch
import numpy as np
from torch import Tensor
from typing import List, Optional
from torch.optim.optimizer import Optimizer, required


def gcsgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        scale:float,
        sparsity:float,
        batch_size:int,
        device):
    
    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf
        #if i == 0:
        val,idx = torch.sort(d_p.flatten(), descending=False)
        torch.where(d_p > val[int(len(val)*sparsity)], d_p, torch.zeros_like(d_p))

            #d_p = d_p + torch.normal(0.0, (scale*C)/math.sqrt(batch_size), d_p.shape).to(device)
            #thresh = np.percentile(d_p.detach().cpu(), 30)
            #thresh = torch.tensor(thresh, dtype=torch.float)
            #print(thresh.dtype, d_p.dtype, type(0.))
            #d_p = torch.where(d_p >= thresh, d_p, 0.).to(device)
        param.add_(d_p, alpha=-lr) 
        #param.add_(torch.normal(0,1,param.shape).to('cuda'))
        #print('=='*20, i, param.shape)




class GCSGD(Optimizer):
    def __init__(self, params, sparsity, scale, batch_size, device, lr=required, 
            momentum=0, dampening=0, weight_decay=0, nesterov=False):
        
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        defaults = dict(lr=lr, sparsity=sparsity, scale=scale, batch_size=batch_size, 
                device=device, momentum=momentum, dampening=dampening,
                weight_decay=weight_decay, nesterov=nesterov)
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        
        super(GCSGD, self).__init__(params, defaults)
        
    def __setstate__(self, state):
        super(GCSGD, self).__setstate__(state)
        
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum'] 
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            #grad_clip = group['grad_clip']
            sparsity = group['sparsity']
            scale = group['scale']
            batch_size = group['batch_size']  
            device = group['device']

            
            for p in group['params']:
                if p.grad is not None:
                    #p.grad = p.grad / torch.max(torch.tensor([1, p.grad.norm()/C]))
                    #p.grad.data.clamp_(-grad_clip, grad_clip)
                    #print(list(p.grad.size()))
                    
                    #p.grad = p.grad + torch.normal(0.0, scale*C, p.grad.shape).to('cuda')
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else: 
                        momentum_buffer_list.append(state['momentum_buffer'])
	    
            gcsgd(params_with_grad,
                    d_p_list,
                    momentum_buffer_list,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    lr=lr,
                    dampening=dampening,
                    nesterov=nesterov,
                    scale=scale,
                    sparsity=sparsity,
                    batch_size=batch_size,
                    device=device)
            
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

