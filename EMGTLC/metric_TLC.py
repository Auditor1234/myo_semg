import torch
import torch.nn.functional as F
import numpy as np
from utils import plot_confusion_matrix

def ACC(output, target, u=None, region_len=100/3, save=False, subject=1, file=None, save_info=[]):
    with torch.no_grad():
        pred = torch.argmax(output,dim=1)
        correct = (pred==target)
        region_correct = (pred/region_len).long()==(target/region_len).long()
    acc = correct.sum().item()/len(target)
    region_acc = region_correct.sum().item()/len(target)
    split_acc = [0,0,0]

    # count number of classes for each region
    num_class = int(3*region_len)
    region_idx = (torch.arange(num_class)/region_len).long()
    region_vol = [
        num_class-torch.count_nonzero(region_idx).item(),
        torch.where(region_idx==1,True,False).sum().item(),
        torch.where(region_idx==2,True,False).sum().item()
    ]
    target_count = target.bincount().cpu().numpy()
    region_vol = [target_count[:region_vol[0]].sum(), target_count[region_vol[0]:(region_vol[0]+region_vol[1])].sum(),target_count[-region_vol[2]:].sum()]
    for i in range(len(target)):
        split_acc[region_idx[target[i].item()]] += correct[i].item()
    split_acc = [split_acc[i]/region_vol[i] for i in range(3)]

    print('Classification ACC:')
    print('\t all \t =',acc)
    print('\t region  =',region_acc)
    print('\t head \t =',split_acc[0])
    print('\t med \t =',split_acc[1])
    print('\t tail \t =',split_acc[2])

    if save and file:
        plot_confusion_matrix(target.cpu().numpy(), pred.cpu().numpy(), 's%d' % subject, file=file)
        num_experts = len(save_info)
        for i in range(num_experts):
            save_info[i] = torch.cat([save_info[i], u.unsqueeze(1), pred.unsqueeze(1), target.unsqueeze(1)], dim=1)
            np.savetxt("res/csv/s%d_expert%din%d_info.csv" % (subject, i, num_experts), save_info[i].cpu().numpy(), delimiter="," ,fmt="%.4f")

    return acc
