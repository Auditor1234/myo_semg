import torch
import torch.nn.functional as F
import torch.nn as nn



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


def edl_mse_loss(output, target, epoch_num=10, classes=10, ecnn_type=2):
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
