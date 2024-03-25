import torch
import copy
import numpy as np
from model import CNN2DEncoder, EMGTLC, ViTEncoder, MoE5, MoE5FC
from train import train
from common_utils import setup_seed, data_label_shuffle
from loss import cross_entropy

from data_process import load_emg_label, filter_labels,\
                            min_max_normal, normal2LT, labels_normal, uniform_distribute, normal2LT


setup_seed(42)

file_fmt = 'datasets/DB5/s%d/repetition%d.pt'
subjects = [1]
long_tail = True
train_rep = [1, 2, 3, 4]
val_rep = [5]
test_rep = [6]
ignore_list = [0, 6]
epochs = 60
num_experts = 4
classes = 10
expert = CNN2DEncoder
save_file = 'res/img/s%d_%dexpert.png' % (subjects[0], num_experts)

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


print(f'-------{num_experts} {expert.__name__}-------')
base_model = expert(classes)
# models = [copy.deepcopy(base_model) for _ in range(num_experts)]
models = [CNN2DEncoder(classes), ViTEncoder(classes), CNN2DEncoder(classes)]

# model = MoE5(models)
model = CNN2DEncoder(classes)
# model = MoE5FC(classes, num_experts)
train(model, epochs, x_train, y_train, x_val, y_val, x_test, y_test, loss_func=cross_entropy)