import torch
import numpy as np
from model import CNN, EMGViT, VCConcat, VCEnsemble, VCEvidential, MoE
from train import train
import matplotlib.pyplot as plt
from common_utils import setup_seed, data_label_shuffle
from data_process import temporal_normal, load_movs_from_file, split_window_ration, load_emg_label_from_file, combine_movs


setup_seed(42)


# subject 2, 6 classification accuracy exceed 80%
subject = 1
file_fmt = 'D:/Download/Datasets/Ninapro/DB2/DB2_s%d/S%d_E1_A1.mat'
movements, labels = load_movs_from_file(file_fmt % (subject, subject))

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

# for i in range(len(emg)):
#     plt.plot(emg[i][:, 0])
#     plt.show()
#     plt.cla()
    
ratio = (4, 1, 1)
window_size=200
window_overlap=100
x_train, y_train, x_val, y_val, test_x, y_test = split_window_ration(emg, labels, ratio=ratio, window_size=window_size, window_overlap=window_overlap)

# from data_process import awgn
# for i in range(len(test_x)):
#     test_x[i] = awgn(test_x[i], snr=20)


x_train = x_train[..., None].transpose((0, 3, 2, 1))
x_val = x_val[..., None].transpose((0, 3, 2, 1))
test_x = test_x[..., None].transpose((0, 3, 2, 1))

x_train, y_train = data_label_shuffle(x_train, y_train)
x_val, y_val = data_label_shuffle(x_val, y_val)
test_x, y_test = data_label_shuffle(test_x, y_test)

mix_subject = 2
mix_ratio = 0.0

# mix_movements, mix_labels = load_movs_from_file(file_fmt % (mix_subject, mix_subject))
# for i in range(len(mix_movements)):
#     mix_movements[i] = np.array(mix_movements[i])[:, :8]
# for i in range(len(mix_movements)):
#     mov_len = len(mix_movements[i])
#     mix_movements[i] = mix_movements[i][int(0.25 * mov_len) : int(0.75 * mov_len)]
# mix_x, mix_y = combine_movs(mix_movements, mix_labels, classes)
# for i in range(len(mix_x)):
#     mix_x[i] = np.vstack(mix_x[i])

mix_x, mix_y = load_emg_label_from_file(file_fmt % (mix_subject, mix_subject))
for i in range(len(mix_x)):
    mix_x[i] = np.array(mix_x[i])[:, :8] * 10000
    mix_x[i] = temporal_normal(mix_x[i])
_, _, _, _, mix_x, mix_y = split_window_ration(mix_x, mix_y, ratio=ratio, window_size=window_size, window_overlap=window_overlap)
mix_x = mix_x[..., None].transpose((0, 3, 2, 1))
mix_x, mix_y = data_label_shuffle(mix_x, mix_y)
mix_num = int(mix_ratio * len(test_x))
for i in range(mix_num):
    test_x[i] = mix_x[i]
    y_test[i] = mix_y[i] - 1

epochs = 60
model_type = 0
models = [EMGViT, CNN, VCConcat, VCEvidential, VCEnsemble]
model = models[model_type](classes)

train(model, epochs, x_train, y_train, x_val, y_val, test_x, y_test, evidential=(model_type == 3))