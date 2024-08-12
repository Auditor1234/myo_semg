import torch
import copy
import numpy as np
from functools import partial
from baseline import TCN, CNN2DEncoder, ECNN_Model
from train import train
# import matplotlib.pyplot as plt
from common_utils import setup_seed, data_label_shuffle
from loss import edl_mse_loss, cross_entropy

from data_process import load_emg_label, normal2LT, uniform_distribute


setup_seed(0)

def main(model, subjects, loss_func=cross_entropy,
            device=torch.device('cuda'),
            dist_path=None):
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
    loss_func = partial(loss_func, classes=classes)
    # save_file = 'res/img/s%d_%dexpert.png' % (subjects[0], 0)
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

    print(f'-------{model.__name__}-------')
    model = model(classes)

    acc, region_acc, split_acc = train(model, epochs, x_train, y_train, x_val, y_val, x_test, y_test, 
                                       loss_func=loss_func,
                                        device=device,
                                       dist_path=dist_path)
    return acc

if __name__ == '__main__':
    models = [TCN, ] # 
    for model in models:
        model_name = model.__name__
        subjects_num = 10
        if 'ECNN' in model_name:
            for ecnn_type in range(3):
                loss_func = partial(edl_mse_loss, ecnn_type=ecnn_type)
                with open(f'res/csv/{model_name}_{ecnn_type}_accuracy.csv', 'w') as f:
                    for subject in range(1, subjects_num + 1):
                        acc = main(model, subjects=subject, loss_func=loss_func)
                        f.write(str(subject) + ',' + '%.2f' % (acc * 100) + '%\n')
                        f.flush()
        else:
            loss_func = cross_entropy
            with open(f'res/csv/{model_name}_accuracy.csv', 'w') as f:
                for subject in range(8, subjects_num + 1):
                    acc = main(model, subjects=subject, loss_func=loss_func)
                    f.write(str(subject) + ',' + '%.2f' % (acc * 100) + '%\n')
                    f.flush()