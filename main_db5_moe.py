import numpy as np
from model import EMGTLC, ViTEncoder, MoE5, MoE5FC, EMGBranchNaive
from train import train
from common_utils import setup_seed, data_label_shuffle
from utils import gen_uncertainty
from loss import cross_entropy, FuseLoss
from config import linear_fc_config, nonlinear2_fc_config, nonlinear3_fc_config
from functools import partial
import torch
from data_process import load_emg_label, filter_labels,\
                            min_max_normal, normal2LT, labels_normal, uniform_distribute, normal2LT



def main(subjects, num_experts, 
         fusion=True, 
         reweight_epoch=30, 
         weight_path='res/weight/best.pt', 
         uncertainty_type='DST',
         device=torch.device('cuda'),
         variable_cloud_size=True,
         dist_path=None,
         ucl_mul=1/2,
         adj_mul=1):
    setup_seed(0)

    file_fmt = 'datasets/DB5/s%d/repetition%d.pt'
    if isinstance(subjects, int):
        subjects = [subjects]
    long_tail = True
    train_rep = [1, 2, 3, 4]
    val_rep = [5]
    test_rep = [6]
    ignore_list = []
    epochs = 60
    classes = 12 - len(ignore_list)
    # save_file = 'res/img/12_s%d_%dexpert.png' % (subjects[0], num_experts)
    save_file = None

    x_train, y_train = load_emg_label(train_rep, file_fmt, subjects, ignore_list)
    x_val, y_val = load_emg_label(val_rep, file_fmt, subjects, ignore_list)
    x_test, y_test = load_emg_label(test_rep, file_fmt, subjects, ignore_list)

    x_train, y_train = np.array(x_train).transpose((0, 2, 1)), np.array(y_train)
    x_val, y_val = np.array(x_val).transpose((0, 2, 1)), np.array(y_val)
    x_test, y_test = np.array(x_test).transpose((0, 2, 1)), np.array(y_test)

    if long_tail:
        x_train, y_train = normal2LT(x_train, y_train)
    # else:
    #     x_train, y_train = uniform_distribute(x_train, y_train)
    x_val, y_val = uniform_distribute(x_val, y_val)
    x_test, y_test = uniform_distribute(x_test, y_test)

    print("train class labels = ", np.bincount(y_train))
    print("val class labels = ", np.bincount(y_val))
    print("test class labels = ", np.bincount(y_test))

    x_train, y_train = data_label_shuffle(x_train, y_train)
    x_val, y_val = data_label_shuffle(x_val, y_val)
    x_test, y_test = data_label_shuffle(x_test, y_test)

    uncertainty_generator = partial(gen_uncertainty, uncertainty_type=uncertainty_type)

    model = EMGBranchNaive(classes, num_experts, 
                           reweight_epoch=reweight_epoch, 
                           fusion=fusion, 
                           gen_uncertainty=uncertainty_generator)
    print(f'-------{num_experts} {model.__class__.__name__}-------')
    subject_config = nonlinear3_fc_config(subjects[0])
    loss_func = FuseLoss(np.bincount(y_train), 
                         reweight_epoch=reweight_epoch,
                         adjust_mul=subject_config.adj_mul,
                         gen_uncertainty=uncertainty_generator,
                         variable_cloud_size=variable_cloud_size,
                         ucl_noise_mul=subject_config.ucl_mul,
                         cloud_size_mul=1)
    acc, region_acc, split_acc = train(model, epochs, x_train, y_train, x_val, y_val, x_test, y_test, 
                                       loss_func=loss_func, 
                                       subject=subjects[0], 
                                       file=save_file,
                                       weight_path=weight_path,
                                       device=device,
                                       dist_path=dist_path)
    return acc, region_acc, split_acc

if __name__ == '__main__':
    subjects = [1]
    num_experts = 1
    main(subjects, num_experts, reweight_epoch=30, uncertainty_type='DST')