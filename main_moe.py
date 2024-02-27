import torch
import numpy as np
from model import CNN, MoE, EMGViT
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

for i in range(len(movements)):
    mov_len = len(movements[i])
    movements[i] = movements[i][int(0.25 * mov_len) : int(0.75 * mov_len)]

classes = 10
emg, labels = combine_movs(movements, labels, classes)

for i in range(len(emg)):
    emg[i] = np.vstack(emg[i]) * 10000
    # emg[i] = temporal_normal(emg[i])


ratio = (4, 1, 1)
window_size=200
window_overlap=100
x_train, y_train, x_val, y_val, x_test, y_test = split_window_ration(emg, labels, ratio=ratio, window_size=window_size, window_overlap=window_overlap)

x_train = x_train[..., None].transpose((0, 3, 2, 1))
x_val = x_val[..., None].transpose((0, 3, 2, 1))
x_test = x_test[..., None].transpose((0, 3, 2, 1))

x_train, y_train = data_label_shuffle(x_train, y_train)
x_val, y_val = data_label_shuffle(x_val, y_val)
x_test, y_test = data_label_shuffle(x_test, y_test)

data_len = len(x_train)
x_train_experts = x_train[:data_len // 2]
y_train_experts = y_train[:data_len // 2]
x_train_moe = x_train[data_len // 2 :]
y_train_moe = y_train[data_len // 2 :]


epochs = 60
models, accuracies = [], []
for i in range(classes):
    mask_expert = (y_train_experts == i) | (y_train_experts == (i + 1) % classes) | (y_train_experts == (i + 2) % classes) \
        | (y_train_experts == (i + 3) % classes) | (y_train_experts == (i + 4) % classes) | (y_train_experts == (i + 5) % classes)
    x_expert = x_train_experts[mask_expert]
    y_expert = y_train_experts[mask_expert]

    expert = EMGViT(classes)
    model, acc = train(expert, epochs, x_expert, y_expert, x_val, y_val, x_test, y_test)
    models.append(model)
    accuracies.append(acc)
    print(f'model {i + 1} done')

plt.plot(accuracies)
plt.show()

moe = MoE(models)
train(moe, epochs, x_train_moe, y_train_moe, x_val, y_val, x_test, y_test)