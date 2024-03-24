import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp
from torch.distributions import normal, dirichlet

class TLCLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, tau=0.5, sigma=0.01):
        super(TLCLoss,self).__init__()

        m_list = 1./np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list*(max_m/np.max(m_list))
        m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
        self.m_list = m_list

        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float)
        frequency_list = torch.log(cls_num_list)
        self.frequency_list = torch.log(sum(cls_num_list)) - frequency_list
        # self.sampler = normal.Normal(0, sigma)
        self.sampler = dirichlet.Dirichlet(self.frequency_list)

        # save diversity per_cls_weights
        self.T = 1000
        self.tau = tau

    def to(self,device):
        super().to(device)
        self.m_list = self.m_list.to(device)
        self.frequency_list = self.frequency_list.to(device)
        return self

    def get_final_output(self, x, y):
        # index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        # y = y.type(torch.int64)
        # index.scatter_(1, y.data.view(-1,1), 1)
        # index_float = index.float()
        # batch_m = torch.matmul(self.m_list[None,:], index_float.transpose(0,1))
        # batch_m = batch_m.view((-1, 1))
        # x_m = x - batch_m * 2
        # return torch.exp(torch.where(index, x_m, x))
        return torch.exp(x)

    def region_loss(self, x, y):
        y = torch.clamp(y - 1, min=0)
        y = (y / 3).long()
        # return self.edl_loss(torch.log, x, y)
        return F.cross_entropy(x, y)
        # return self.loglikelihood_loss(x, y)
    
    def loglikelihood_loss(self, x, y):
        alpha = x + 1
        y = F.one_hot(y, num_classes=3)
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        loglikelihood = loglikelihood_err + loglikelihood_var
        return loglikelihood.mean()
    
    def edl_loss(self, func, x, y):
        alpha = x + 1
        y = F.one_hot(y, num_classes=3)
        S = torch.sum(alpha, dim=1, keepdim=True)
        A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
        return A.mean()

    def forward(self,x,y,epoch,extra_info=None):
        loss = 0.0
        for i in range(extra_info["num_expert"]):
            alpha = self.get_final_output(extra_info["logits"][i], y)

            # viariation = self.sampler.sample(alpha[:, 0].shape).clamp(-1, 1).to(alpha.device)
            # alpha = alpha + viariation.abs() / self.frequency_list.max() * self.frequency_list / self.T

            S = alpha.sum(dim=1,keepdim=True)
            y = y.type(torch.int64)
            l = F.nll_loss(torch.log(alpha)-torch.log(S),y,reduction="none")

            # KL
            yi = F.one_hot(y,num_classes=alpha.shape[1])

            # adjusted parameters of D(p|alpha)
            alpha_tilde = yi+(1-yi)*(alpha+1)
            S_tilde = alpha_tilde.sum(dim=1,keepdim=True)
            kl = torch.lgamma(S_tilde)-torch.lgamma(torch.tensor(alpha_tilde.shape[1]))-torch.lgamma(alpha_tilde).sum(dim=1,keepdim=True) \
                +((alpha_tilde-1)*(torch.digamma(alpha_tilde)-torch.digamma(S_tilde))).sum(dim=1,keepdim=True)
            l += epoch/self.T*kl.squeeze(-1)

            # diversity
            output_dist = F.log_softmax(extra_info["logits"][i],dim=1)
            with torch.no_grad():
                mean_output_dist = F.softmax(x,dim=1)
            l -= 0.01*F.kl_div(output_dist,mean_output_dist,reduction="none").sum(dim=1)

            # dynamic engagement
            w = extra_info['w'][i]
            loss += l.sum() / len(l)
            w = torch.where(w>self.tau,True,False)
            if w.sum() != 0:
                loss += (w*l).sum()/w.sum()

        return loss