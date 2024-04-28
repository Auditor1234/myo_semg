import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, title, labels_name=None, colorbar=False, file='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=None)
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    if colorbar:
        plt.colorbar()
    if labels_name == None:
        labels_name = ['%d' % i for i in range(len(cm))]
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.title(title)    # 图像标题
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
    plt.savefig(file, bbox_inches='tight')


def ACC(output, target, region_len=100/3, subject=1, file=None, save_info=[]):
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

    
    if file:
        plot_confusion_matrix(target.cpu().numpy(), pred.cpu().numpy(), 's%d' % subject, file=file)
        # num_experts = len(save_info)
        # for i in range(num_experts):
        #     save_info[i] = torch.cat([save_info[i], pred.unsqueeze(1), target.unsqueeze(1)], dim=1)
        #     np.savetxt("res/csv/s%d_expert%din%d_info.csv" % (subject, i, num_experts), save_info[i].cpu().numpy(), delimiter="," ,fmt="%.4f")

    return acc
