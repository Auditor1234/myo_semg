import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.distributions import normal
from utils import gen_uncertainty



def combine_loss(evidences, y, epoch_num=10, classes=10):
    def KL(alpha, c): # shape(B,c)
        beta = torch.ones((1, c)).cuda() # shape(1,c)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True) # shape(B,1)
        S_beta = torch.sum(beta, dim=1, keepdim=True) # shape(1,1)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True) # shape(B,1)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha) # shape(B,1)
        dg1 = torch.digamma(alpha) # shape(B,c)
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl # shape(B,1)


    def ce_loss(p, alpha, c, global_step, annealing_step):
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        label = F.one_hot(p, classes=c) # shape(B,c)
        A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

        annealing_coef = min(1, global_step / annealing_step)

        alp = E * (1 - label) + 1
        B = annealing_coef * KL(alp, c)

        return (A + B) # shape(B,1)


    def DS_Combin(alpha, classes): # shape(V,B,c)
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True) # shape(B,1)
                E[v] = alpha[v]-1 # shape(B,c)
                b[v] = E[v]/(S[v].expand(E[v].shape)) # shape(B,c)
                u[v] = classes/S[v] # shape(B,1)
            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape) # shape(B,c)
            bu = torch.mul(b[0], uv1_expand) # shape(B,c)点乘，哈达玛积
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag # shape(B,1)
            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape)) # shape(B,c)
            # calculate u^a
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape)) # shape(B,1)
            # calculate new S
            S_a = classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a
        for v in range(len(alpha)-1):
            if v==0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a

    loss = 0
    alpha = dict()
    for v_num in range(len(evidences)):
        alpha[v_num] = evidences[v_num] + 1
        loss += ce_loss(y, alpha[v_num], classes, 1, 1)
    alpha_a = DS_Combin(alpha, classes=classes)
    evidence_a = alpha_a - 1
    loss += ce_loss(y, alpha_a, classes, 1, 1)
    loss = torch.mean(loss)
    return loss


def cross_entropy(predictions, y, epoch_num=10, classes=10):
    return F.cross_entropy(predictions, y)


def edl_mse_loss(output, target, epoch_num=10, classes=12, ecnn_type=2):
    """
    ecnn_type: 0 for ECNN-A, 1 for ECNN-B and 2 for ECNN-C
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def kl_divergence(alpha):
        beta = torch.ones([1, classes], dtype=torch.float32, device=device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                            keepdim=True) - torch.lgamma(S_beta)

        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)

        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                       keepdim=True) + lnB + lnB_uni
        return kl
    
    def loglikelihood_loss(y, alpha):
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum(
            (y - (alpha / S)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        loglikelihood = loglikelihood_err + loglikelihood_var
        return loglikelihood

    def mse_loss(y, alpha):
        y = y.to(device)
        alpha = alpha.to(device)
        loglikelihood = loglikelihood_loss(y, alpha)
        if ecnn_type == 0:
            return loglikelihood
        elif ecnn_type == 1:
            annealing_coef = \
                torch.min(
                    torch.tensor(1.0, dtype=torch.float32, device=device),
                    torch.tensor(epoch_num / 10, dtype=torch.float32, device=device)
                    )
            kl_alpha = (alpha - 1) * (1 - y) + 1
            kl_div = annealing_coef * kl_divergence(kl_alpha)
            return loglikelihood + kl_div  # + (1-p_t)*kl_div
        else:
            kl_alpha = (alpha - 1) * (1 - y) + 1
            coef = torch.tensor(0.1, dtype=torch.float32)
            kl_div = coef * kl_divergence(kl_alpha)
            return loglikelihood + kl_div


    evidence = F.relu(output)
    alpha = evidence + 1
    y = F.one_hot(target, num_classes=classes)
    loss = mse_loss(y.float(), alpha)
    return sum(loss) / len(loss)



class FuseLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, reweight_epoch=30, reweight_factor=0.05, adjust_mul=1,
                 gen_uncertainty=None, variable_cloud_size=False,
                 ucl_noise_mul=1/3, cloud_size_mul=5/3):
        super(FuseLoss,self).__init__()
        self.reweight_epoch = reweight_epoch

        m_list = 1./np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list*(max_m/np.max(m_list))
        m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
        self.m_list = m_list
        self.num_cls = len(cls_num_list)

        # uncertainty-based cloud learning
        self.adjust_mul = adjust_mul
        self.gen_uncertainty = gen_uncertainty
        self.variable_cloud_size = variable_cloud_size
        self.gcl_noise_mul = 1 / 2
        self.ucl_noise_mul = ucl_noise_mul # -1 / 3
        self.cloud_size_mul = cloud_size_mul # 5 / 3
        gcl_m_list = torch.log(torch.tensor(cls_num_list))
        gcl_m_list = gcl_m_list.max()-gcl_m_list
        self.gcl_m_list = gcl_m_list


        if reweight_epoch!=-1:
            idx = 1
            betas = [0,0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights_enabled = None
        cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
        C = len(cls_num_list)
        per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor
        per_cls_weights = per_cls_weights / np.max(per_cls_weights)

        # save diversity per_cls_weights
        self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights,dtype=torch.float,requires_grad=False)
        self.per_cls_weights_diversity = None
        self.tempature = []

    def to(self,device):
        super().to(device)
        self.m_list = self.m_list.to(device)
        self.gcl_m_list = self.gcl_m_list.to(device)
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)
        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)
        return self

    def _hook_before_epoch(self,epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch
            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self,x,y):
        index = torch.zeros_like(x,dtype=torch.uint8,device=x.device)
        y = y.type(torch.int64)
        index.scatter_(1,y.data.view(-1,1),1)
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None,:],index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x-30*batch_m
        return torch.exp(torch.where(index,x_m,x))

    def forward(self, x, y, epoch, extra_info=None):
        device = x.device
        loss = 0.0
        for i in range(extra_info["num_expert"]):

            y = y.type(torch.int64)
            index = torch.zeros_like(x,dtype=torch.uint8,device=x.device)
            index.scatter_(1,y.data.view(-1,1),1)

            # logits adjustment
            logits = extra_info["logits"][i]
            
            scale = torch.mean(torch.abs(logits), dim=-1, keepdim=True)
            if self.variable_cloud_size:
                alpha = torch.where(index, logits - scale * self.m_list * self.adjust_mul, logits)
                # if epoch < self.reweight_epoch:
                #     l = F.cross_entropy(alpha, y, weight=self.per_cls_weights_base)
                # else:
                # uncertainty-aware perturbation learning
                with torch.no_grad():
                    cloud_size = self.gen_uncertainty(logits).to(device)
                    cloud_size = torch.clamp(cloud_size, min=1 / 4) # 1 / 3
                sampler = normal.Normal(torch.zeros(x.shape[:1], device=device),  self.cloud_size_mul * cloud_size)
                variation = self.ucl_noise_mul * sampler.sample(x.shape[1:]).to(device).permute(1, 0).clamp(-1, 1).abs()
                variation = variation * scale
                l = F.cross_entropy(alpha - variation, y, weight=self.per_cls_weights_base)

                # # adjusted parameters of D(p|alpha)
                # yi = F.one_hot(y,num_classes=alpha.shape[1])
                # alpha = torch.exp(logits.clamp(max=3))
                # alpha_tilde = yi+(1-yi)*(alpha+1)
                # l += min(1, (epoch + 1) / 30) * KL(alpha_tilde, self.num_cls).mean()

                # # diversity
                # if self.per_cls_weights_diversity is not None:
                #     diversity_temperature = self.per_cls_weights_diversity.view((1,-1))
                #     temperature_mean = diversity_temperature.mean().item()
                # else:
                #     diversity_temperature = 1
                #     temperature_mean = 1
                # output_dist = F.log_softmax(logits/diversity_temperature,dim=1)
                # with torch.no_grad():
                #     mean_output_dist = F.softmax(x/diversity_temperature,dim=1)
                # l -= 0.01*temperature_mean*temperature_mean*F.kl_div(output_dist,mean_output_dist,reduction="none").sum(dim=1).mean()
            else:
                cloud_size = 1 / 3
                sampler = normal.Normal(torch.zeros(x.shape[:1], device=device),  cloud_size)
                variation = self.gcl_noise_mul * sampler.sample(x.shape[1:]).to(device).permute(1, 0).clamp(-1, 1).abs() \
                    / self.gcl_m_list.max() * self.gcl_m_list
                variation = variation * scale
                logits = torch.where(index, logits - variation, logits)
                l = F.cross_entropy(logits , y, weight=self.per_cls_weights_base)

            loss += l.mean()
        return loss

class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = nn.Parameter(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num =  class_num

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1) # softmmax获取预测概率
        class_mask = F.one_hot(target, self.class_num) #获取target的one hot编码
        ids = target.view(-1, 1) 
        alpha = self.alpha[ids.data.view(-1)] # 注意，这里的alpha是给定的一个list,里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        log_p = probs.log()
        # 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

def focal_loss(predict, target, gamma=2):
    pt = F.softmax(predict, dim=1) # softmmax获取预测概率
    class_mask = F.one_hot(target, predict.shape[1]) #获取target的one hot编码
    probs = (pt * class_mask).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
    log_p = probs.log()
    loss = -(torch.pow((1 - probs), gamma)) * log_p
    return loss.mean()


def KL(alpha, c):
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    beta = torch.ones((1, c)).cuda()
    # Mbeta = torch.ones((alpha.shape[0],c)).cuda()
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl
