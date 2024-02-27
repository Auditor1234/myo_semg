import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp

class TLCLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, tau=0.05):
        super(TLCLoss,self).__init__()

        m_list = 1./np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list*(max_m/np.max(m_list))
        m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
        self.m_list = m_list

        # save diversity per_cls_weights
        self.T = 100
        self.tau = tau

    def to(self,device):
        super().to(device)
        self.m_list = self.m_list.to(device)
        return self

    def get_final_output(self, x, y):
        index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        y = y.type(torch.int64)
        index.scatter_(1, y.data.view(-1,1), 1)
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None,:], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * 2
        return torch.exp(torch.where(index, x_m, x))

    def forward(self,x,y,epoch,extra_info=None):
        loss = 0.0
        for i in range(extra_info["num_expert"]):
            alpha = self.get_final_output(extra_info["logits"][i], y)
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
            w = torch.where(w>self.tau,True,False)
            if w.sum() != 0:
                loss += (w*l).sum()/w.sum()

        return loss