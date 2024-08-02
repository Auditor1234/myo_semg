import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import umap
import torch.nn.functional as F


def plot_confusion_matrix(y_true, y_pred, title, labels_name=None, colorbar=False, file='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.cla()
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
    print('\t all \t = %.2f' % (acc * 100) + '%')
    print('\t region  = %.2f' % (region_acc * 100) + '%')
    print('\t head \t = %.2f' % (split_acc[0] * 100) + '%')
    print('\t med \t = %.2f' % (split_acc[1] * 100) + '%')
    print('\t tail \t = %.2f' % (split_acc[2] * 100) + '%')

    
    if file:
        plot_confusion_matrix(target.cpu().numpy(), pred.cpu().numpy(), 's%d' % subject, file=file)
        # num_experts = len(save_info)
        # for i in range(num_experts):
        #     save_info[i] = torch.cat([save_info[i], pred.unsqueeze(1), target.unsqueeze(1)], dim=1)
        #     np.savetxt("res/csv/s%d_expert%din%d_info.csv" % (subject, i, num_experts), save_info[i].cpu().numpy(), delimiter="," ,fmt="%.4f")

    return acc, region_acc, split_acc

def plot_features(x, y, save_path='res/img/features.png', mode='tsne'):
    if mode == 'tsne':
        reducer = TSNE(random_state=42)
    else:
        reducer = umap.UMAP(random_state=42)
    results = reducer.fit_transform(x)
    plt.clf()
    plt.scatter(results[:, 0], results[:, 1], c=y, label=mode)
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.legend()
    plt.savefig(save_path, dpi=120)

def gen_uncertainty(logits, uncertainty_type='DST'):
    output = F.normalize(logits.clone().detach())
    uncertainty = None
    if uncertainty_type == 'DST':
        alpha = torch.exp(output) + 1
        S = alpha.sum(dim=1,keepdim=True)
        uncertainty = output.shape[1] / S.squeeze(-1)
    elif uncertainty_type == 'RSM':
        output_exp = torch.exp(output)
        max_val, max_idx = torch.max(output_exp, dim=-1)
        row_idx = torch.arange(output_exp.shape[0])
        
        non_target_logits = torch.ones_like(output_exp)
        non_target_logits[row_idx, max_idx] = 0
        non_target_logits *= output_exp
        
        second_val, _ = torch.max(non_target_logits, dim=-1)
        uncertainty = second_val / max_val
    else:
        raise Exception('uncertaint type can only be DST or RSM.')
    return uncertainty

def plot_uncertainty_accuracy(output, y_true):
    output = output.cpu()
    y_pred = torch.argmax(output, dim=1)
    y_pred = y_pred.numpy()
    y_true = y_true.cpu().numpy()
    num_class = output.shape[1]

    cm = confusion_matrix(y_true, y_pred)
    acc = []
    for i in range(len(cm)):
        acc.append(cm[i, i] / sum(cm[:, i]))
    
    uncertainty = gen_uncertainty(output)
    u_avg = []
    for i in  range(num_class):
        u_i = uncertainty[y_true == i]
        u_avg.append(u_i.mean().numpy())


    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(acc, label='accuracy', color='green', linestyle='-',marker='o')
    ax1.set_ylabel("accuracy")

    ax2 = ax1.twinx()
    ax2.plot(u_avg, label='uncertainty', color='red', linestyle='-',marker='X')
    ax2.set_ylabel("uncertainty", labelpad=40)

    fig.legend(loc="upper right")
    plt.title('uncertainty vs accuracy')
    plt.savefig('res/img/uncertainty-accuracy.png')

    