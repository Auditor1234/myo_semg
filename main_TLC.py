import torch
import copy
import numpy as np
from model import CNN, EMGTLC, EMGViT
from EMGTLC.train_TLC import train
import matplotlib.pyplot as plt
from common_utils import setup_seed, data_label_shuffle

from data_process import temporal_normal, load_movs_from_file, split_window_ration, tenfold_augmentation,\
                            load_emg_label_from_file, combine_movs, normal2LT, labels_normal, uniform_distribute


setup_seed(42)

# subject 2, 6 classification accuracy exceed 80%
subject = 1
file_fmt = 'D:/Download/Datasets/Ninapro/DB2/DB2_s%d/S%d_E1_A1.mat'
movements, labels = load_movs_from_file(file_fmt % (subject, subject))
labels = labels_normal(labels)

for i in range(len(movements)):
    movements[i] = np.array(movements[i])[:, :8]

# for i in range(len(movements)):
#     mov_len = len(movements[i])
#     movements[i] = movements[i][int(0.25 * mov_len) : int(0.75 * mov_len)]

classes = 10
emg, labels = combine_movs(movements, labels, classes)

for i in range(len(emg)):
    emg[i] = np.vstack(emg[i]) * 10000
    # emg[i] = temporal_normal(emg[i])

descend_idx = np.argsort([len(emg[i]) for i in range(len(emg))])[::-1]
emg_temp = []
for i in range(len(descend_idx)):
    emg_temp.append(emg[descend_idx[i]])
emg = emg_temp
    
ratio = (4, 1, 1)
window_size=200
window_overlap=100
x_train, y_train, x_val, y_val, x_test, y_test = split_window_ration(emg, labels, ratio=ratio, window_size=window_size, window_overlap=window_overlap)

x_train, y_train = normal2LT(x_train, y_train)
x_val, y_val = uniform_distribute(x_val, y_val)
x_test, y_test = uniform_distribute(x_test, y_test)

x_train, y_train = tenfold_augmentation(x_train, y_train)
x_val, y_val = tenfold_augmentation(x_val, y_val)
x_test, y_test = tenfold_augmentation(x_test, y_test)

print("train class labels = ", np.bincount(y_train))
print("test class labels = ", np.bincount(y_test))
# from data_process import awgn
# for i in range(len(x_test)):
#     x_test[i] = awgn(x_test[i], snr=20)


x_train = x_train[..., None].transpose((0, 3, 2, 1))
x_val = x_val[..., None].transpose((0, 3, 2, 1))
x_test = x_test[..., None].transpose((0, 3, 2, 1))

x_train, y_train = data_label_shuffle(x_train, y_train)
x_val, y_val = data_label_shuffle(x_val, y_val)
x_test, y_test = data_label_shuffle(x_test, y_test)

epochs = 60
num_experts = 2
expert = CNN
print(f'-------{num_experts} {expert.__name__}-------')
base_model = expert(classes)
model = EMGTLC(
    [copy.deepcopy(base_model) for _ in range(num_experts)]
)

train(model, epochs, x_train, y_train, x_val, y_val, x_test, y_test)