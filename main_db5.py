import torch
import copy
import numpy as np
from model import CNN2DEncoder, EMGTLC, ViTEncoder
from EMGTLC.train_TLC import train
import matplotlib.pyplot as plt
from common_utils import setup_seed, data_label_shuffle
from loss import cross_entropy

from data_process import load_movs_from_file, split_window_ration, tenfold_augmentation, filter_labels,\
                            min_max_normal, normal2LT, labels_normal, uniform_distribute, normal2LT


setup_seed(42)

file_fmt = 'datasets/DB5/s%d/repetition%d.pt'
subject = 1
long_tail = True
train_rep = [1, 2, 3, 4]
val_rep = [5]
test_rep = [6]
ignore_list = [0, 6]
epochs = 60
num_experts = 1
classes = 10
expert = CNN2DEncoder
save_file = 'res/s%d_%dexpert.png' % (subject, num_experts)

x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
for rep in train_rep:
    data = torch.load(file_fmt % (subject, rep))
    rep_emg = data['emg']
    rep_label = data['label']
    for i in range(len(rep_label)):
        x_train.append(rep_emg[i])
        y_train.append(rep_label[i])

for rep in val_rep:
    data = torch.load(file_fmt % (subject, rep))
    rep_emg = data['emg']
    rep_label = data['label']
    for i in range(len(rep_label)):
        x_val.append(rep_emg[i])
        y_val.append(rep_label[i])

for rep in test_rep:
    data = torch.load(file_fmt % (subject, rep))
    rep_emg = data['emg']
    rep_label = data['label']
    for i in range(len(rep_label)):
        x_test.append(rep_emg[i])
        y_test.append(rep_label[i])

x_train, y_train = filter_labels(x_train, y_train, ignore_list)
x_val, y_val = filter_labels(x_val, y_val, ignore_list)
x_test, y_test = filter_labels(x_test, y_test, ignore_list)

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
print("test class labels = ", np.bincount(y_test))

x_train, y_train = data_label_shuffle(x_train, y_train)
x_val, y_val = data_label_shuffle(x_val, y_val)
x_test, y_test = data_label_shuffle(x_test, y_test)


print(f'-------{num_experts} {expert.__name__}-------')
base_model = expert(classes)
model = EMGTLC(
    [copy.deepcopy(base_model) for _ in range(num_experts)]
)

train(model, epochs, x_train, y_train, x_val, y_val, x_test, y_test, subject, file=save_file)