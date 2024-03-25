import h5py
import math
import glob
import torch
import numpy as np
import scipy.io as scio
from feature_utils import *
from scipy.signal import butter, sosfilt, filtfilt
from pathlib import Path
from numpy.random import pareto


def load_emg_label_from_dir(file):
    """
    param:
    ---
        file: mat file or mat files in list
    """
    files = file if isinstance(file, list) else [file]
    
    emg, label = [], []
    # iterate each file
    for file in files:
        mat_file = scio.loadmat(file)
        file_emg = mat_file['emg']
        file_label = mat_file['restimulus']

        # store one file data except 'rest' action
        temp_emg, temp_label = [], []
        for i in range(len(file_label)):
            if file_label[i] == 0:
                continue
            temp_emg.append(file_emg[i])
            temp_label.append(file_label[i])
        print('{} data has read, get {} valid points.'.format(file, len(temp_label)))
        emg += temp_emg
        label += temp_label

    emg = np.vstack(emg)
    label = np.vstack(label)
    label = label - 1

    print('emg.shape = ', emg.shape)
    print('label.shape = ', label.shape)

    return emg, label


def feature2h5py(emg, label, dest_file):
    print('\n{} different classes.'.format(len(np.unique(label))))
    featureData=[]
    featureLabel = []
    classes = 48
    timeWindow = 200
    strideWindow = 200

    for i in range(classes):
        index = []
        for j in range(label.shape[0]):
            if label[j, :] == i:
                index.append(j)
        iemg = emg[index,:]
        length = math.floor(iemg.shape[0] / strideWindow)
        print("class ", i, ", number of sample: ", iemg.shape[0], length)

        for j in range(length):
            rms = featureRMS(iemg[strideWindow*j : strideWindow*j + timeWindow, :])
            mav = featureMAV(iemg[strideWindow*j : strideWindow*j + timeWindow, :])
            wl  = featureWL( iemg[strideWindow*j : strideWindow*j + timeWindow,:])
            zc  = featureZC( iemg[strideWindow*j : strideWindow*j + timeWindow, :])
            ssc = featureSSC(iemg[strideWindow*j : strideWindow*j + timeWindow, :])
            featureStack = np.hstack((rms,mav,wl,zc,ssc))

            featureData.append(featureStack)
            featureLabel.append(i)

    featureData = np.array(featureData)
    print('featureData.shape = ', featureData.shape)
    print('featureLabel.shape = ', len(featureLabel))

    file = h5py.File(dest_file,'w')  
    file.create_dataset('featureData', data = featureData)  
    file.create_dataset('featureLabel', data = featureLabel)  
    file.close()  


def image2h5py(emg, label, dest_file):
    print('\n{} different classes.'.format(len(np.unique(label))))
    emg = emg*20000

    imageData=[]
    imageLabel=[]
    imageLength=200
    classes = 49

    for i in range(classes):
        index = []
        for j in range(label.shape[0]):
            if label[j, :] == i:
                index.append(j)

        iemg = emg[index,:]
        length = math.floor(iemg.shape[0] / imageLength)
        print("class", i, " number of sample: ",iemg.shape[0], length)

        for j in range(length):
            subImage = iemg[imageLength*j : imageLength*(j + 1), :]
            imageData.append(subImage)
            imageLabel.append(i)
        
    imageData = np.array(imageData)
    print('imageData.shape =', imageData.shape)
    print('imageLabel.shape =', len(imageLabel))

    file = h5py.File(dest_file,'w')  
    file.create_dataset('imageData', data = imageData)  
    file.create_dataset('imageLabel', data = imageLabel)  
    file.close()


def load_movs_from_file(filename):
    """
    load .mat file from input filename
    """
    mat_file = scio.loadmat(filename)
    
    emg = mat_file['emg'] # shape(877073, 12)
    restimulus = mat_file['restimulus'] # shape(877072, 1)
    rerepetition = mat_file['rerepetition'] # shape(877072, 1)
    # acc = mat_file['acc'] # shape(877073, 36)
    # force = mat_file['force'] # shape(877073, 6)
    # forcecal = mat_file['forcecal'] # shape(2, 6)
    # activation = mat_file['activation'] # shape(877073, 6)

    movements = []
    labels = []

    print("{} data points in file {} found.".format(len(emg), filename.replace('\\', '/')))
    for i in range(len(emg) - 1):
        if rerepetition[i] < 1:
            continue
        if i == 0 or restimulus[i] != restimulus[i - 1]:
            movements.append([])
            labels.append(restimulus[i][0])
        else:
            movements[-1].append(emg[i].tolist()) # shape(*, 48)
    print("Data in {} loaded.".format(filename.replace('\\', '/')))
    return movements, labels


def load_movs_from_dir(path):
    """
    load .mat file from input directory path
    """
    files = glob.glob(f"{path}/*/*.mat")
    if len(files) == 0:
        print("Error: no dataset found!")
        exit(1)

    total_movements, total_labels = [], []
    print("Loading data...")
    for file in files:
        movements, labels = load_movs_from_file(file)
        total_movements += movements
        total_labels += labels
    print("{} movements have been loaded.".format(len(total_labels)))
    return total_movements, total_labels


def combine_movs(movements, labels, classes=10):
    emg = [[] for _ in range(classes)]
    for label in np.unique(labels):
        if label == 0 or label > classes:
            continue
        for i in range(len(labels)):
            if labels[i] == label:
                emg[label - 1].append(movements[i])
    
    return emg, list(range(classes))


def load_emg_label_from_file(filename, class_type=10):
    emg, label = [], []
    for i in range(class_type):
        emg.append([])

    # iterate each file
    mat_file = scio.loadmat(filename)
    file_emg = mat_file['emg']
    file_label = mat_file['restimulus']


    # store one file data except 'rest' action
    for i in range(len(file_label)):
        label_idx = file_label[i][0]
        if label_idx == 0 or label_idx > class_type:
            continue
        movement_idx = label_idx - 1
        if len(emg[movement_idx]) == 0:
            label.append(label_idx)
        emg[movement_idx].append(file_emg[i].tolist())
    print('{} has read, get {} types movement.'.format(filename, class_type))


    print('emg.length = ', len(emg))
    print('label = ', label)

    return emg, label
    

def window_to_h5py(emg, label, filename, window_size=400, window_overlap=0):
    window_data = []
    window_label = []
    for i in range(len(label)):
        emg_type = np.array(emg[i])
        window_count = 0
        print('{} emg points found in type {} emg signal.'.format(len(emg_type), label[i]))
        for j in range(0, len(emg_type) - window_size, window_size - window_overlap):
            window_data.append(emg_type[j : j + window_size])
            window_label.append(label[i])
            window_count += 1
        print('{} window data found in type {} emg signal.'.format(window_count, label[i]))
    
    file = h5py.File(filename,'w')  
    file.create_dataset('windowData', data = np.stack(window_data, axis=0))
    file.create_dataset('windowLabel', data = np.array(window_label))
    file.close()


def h5py_to_window(filename):
    file = h5py.File(filename, 'r')
    emg = file['windowData'][:]
    label = file['windowLabel'][:]
    file.close()
    return emg, label


def split_window_ration(emg, label, ratio, window_size=400, window_overlap=200):
    """
    emg: shape(points, channels)
    """
    denominator = sum(ratio)

    train_emg, train_label, val_emg, val_label, eval_emg, eval_label = [], [], [], [], [], []

    for i in range(len(label)):
        data_len = len(emg[i])
        train_len = int(data_len * ratio[0] / denominator)
        val_len = int(data_len * ratio[1] / denominator)

        emg_train = np.array(emg[i][:train_len])
        window_count = 0
        for j in range(0, len(emg_train) - window_size, window_size - window_overlap):
            train_emg.append(emg_train[j : j + window_size])
            train_label.append(label[i])
            window_count += 1
        # print('{} train window data found in type {} emg signal.'.format(window_count, label[i]))

        emg_val = np.array(emg[i][train_len : train_len + val_len])
        window_count = 0
        for j in range(0, len(emg_val) - window_size, window_size):
            val_emg.append(emg_val[j : j + window_size])
            val_label.append(label[i])
            window_count += 1
        # print('{} val window data found in type {} emg signal.'.format(window_count, label[i]))

        emg_eval = np.array(emg[i][train_len + val_len :])
        window_count = 0
        for j in range(0, len(emg_eval) - window_size, window_size):
            eval_emg.append(emg_eval[j : j + window_size])
            eval_label.append(label[i])
            window_count += 1
        # print('{} eval window data found in type {} emg signal.'.format(window_count, label[i]))

    train_emg = np.array(train_emg)
    train_label = np.array(train_label)
    val_emg = np.array(val_emg)
    val_label = np.array(val_label)
    eval_emg = np.array(eval_emg)
    eval_label = np.array(eval_label)
    return train_emg, train_label, val_emg, val_label, eval_emg, eval_label


def filter_signals(signal, fs):
    """
    Extracts the envelopes of the multi-channel sEMG signal as described in [1]
    :param signal: The multi channel sEMG signal, shape = (no_channels, no_samples)
    :return: the signal envelopes of the multi-channel signals, shape = (no_channels, no_samples)
    """
    signal = np.abs(signal)  # full wave rectification
    lpf = butter(2, 0.8, 'lowpass', analog=False, fs=fs,
                 output='sos')  # define the low pass filter (Butterworth 2-order fc = 1Hz)
    filtered_signal = sosfilt(lpf, signal)
    return filtered_signal


def get_tma_maps(signal, obs_inc=0.2, obs_dur=0.2, fs=2000, origin=False):
    i = 0
    data_len = int(obs_dur * fs)
    action_evolution_maps = []

    while True:
        obs_start = int(obs_inc * fs * i)
        obs_end = int(obs_inc * fs * i + obs_dur * fs)
        time = (obs_start + obs_end) / (2.0 * fs)
        if obs_end > signal.shape[1]:
            break
        obs = signal[:, obs_start:obs_end]
        # U = obs
        if origin:
            action_evolution_maps.append(obs)
        else:
            U = non_linear_transform(obs, data_len=data_len)
            action_evolution_maps.append(U)
        i += 1
    return np.stack(action_evolution_maps)


def non_linear_transform(X, data_len):
    """
    Generates the TMA map for the multi-channel signals of the
    specified time window using the non-linear transform described in [1].
    :param X: the multi-channel signals of the
    specified time window
    :return: the TMA map for the multi-channel signals of the specified time window
    """
    U = np.zeros((44, data_len))  # define the TMA map
    for t in range(data_len):
        ut = X[:, t]
        ut = ut.reshape(8, 1)
        cov = ut * ut.T  # obtain the covariance matrix
        idx = np.triu_indices(8)
        temp = np.concatenate((np.squeeze(ut), cov[idx]))  # removed 1
        U[:, t] = temp
    U1 = U[:8, :]
    U[:8, :] = (U1 - U1.min()) / (U1.max() - U1.min())
    return U


def lpf(x, f=1., fs=100):
    f = f / (fs / 2)
    x = np.abs(x)
    b, a = butter(1, f, 'low')
    output = filtfilt(
        b, a, x, axis=0,
        padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)
    )
    return output


def min_max_normal(emg, dim=0, eps=1e-6):
    """
    min max normalization
    """
    diff = np.max(emg, axis=dim, keepdims=True) - np.min(emg, axis=dim, keepdims=True)
    return (emg - np.min(emg, axis=dim, keepdims=True)) / (diff + eps)


def temporal_normal(emg, dim=0, eps=1e-6):
    """
    temporal min max normalization
    """
    emg_min = np.min(emg[:1], axis=dim, keepdims=True)
    emg_max = np.max(emg[:1], axis=dim, keepdims=True)

    for i in range(len(emg)):
        emg_min = np.min((emg_min, emg[i : i + 1]), axis=dim)
        emg_max = np.max((emg_max, emg[i : i + 1]), axis=dim)
        emg[i] = (emg[i] - emg_min) / (emg_max - emg_min + eps)
    return emg


def awgn(x, snr):
    '''
    加入高斯白噪声 Additive White Gaussian Noise
    :param x: 原始信号, shape(window_size, channels)
    :param snr: 信噪比
    :return: 加入噪声后的信号
    '''
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2, axis=0) / len(x)
    npower = xpower / snr
    noise = np.random.randn(x.shape[0], x.shape[1]) * np.sqrt(npower)
    return x + noise

def normal2LT(emg, labels, mu=0.65):
    """
    construct long-tailed data
    """
    label_bins = np.bincount(labels)
    emg_type = [[] for _ in range(len(label_bins))]
    for i in range(len(labels)):
        emg_type[labels[i]].append(emg[i])
    
    max_num = max(label_bins)
    LT_ratio = [mu ** i for i in range(0, len(label_bins))]

    labels_out = []
    emg_out = []
    for i in range(len(LT_ratio)):
        sample_idx = np.random.choice(len(emg_type[i]), min(len(emg_type[i]), int(max_num * LT_ratio[i])), replace=False)
        for j in range(len(sample_idx)):
            emg_out.append(emg_type[i][sample_idx[j]])
            labels_out.append(i)
            # if i == 8 and j == 19:
            #     break
            # if i == 9 and j == 9:
            #     break
    return np.stack(emg_out), np.stack(labels_out)

def labels_normal(labels):
    unique_labels = sorted(np.unique(labels))
    labels_dict = {unique_labels[i] : i for i in range(len(unique_labels))}
    for i in range(len(labels)):
        labels[i] = labels_dict[labels[i]]
    return labels

def uniform_distribute(emg, labels):
    """
    make uniform distribution data
    """
    min_nums = min(np.bincount(labels))
    emg_temp, labels_temp = [], []
    for _ in range(len(np.unique(labels))):
        emg_temp.append([])
    for i in range(len(labels)):
        if len(emg_temp[labels[i]]) < min_nums:
            emg_temp[labels[i]].append(emg[i])
            labels_temp.append(labels[i])
    return np.array(emg_temp).reshape((-1, emg.shape[1], emg.shape[2])), np.array(labels_temp)
    
def tenfold_augmentation(emg, labels):
    """
    ten times data augmentation
    """
    emg = np.repeat(emg, repeats=10, axis=0)
    labels = np.repeat(labels, repeats=10, axis=0)
    return emg, labels


def sliding_window(arr, window_length, stride):
    shape = (window_length, *arr.shape[1:])
    return np.lib.stride_tricks.sliding_window_view(arr, shape)[::stride].squeeze()

def prepare_data(input_dir, output_dir, exercise, subjects, classes, repetitions, window_length, stride):
    for s in subjects:
        path = Path(f'{output_dir}/s{s}')
        path.mkdir(parents=True, exist_ok=True)
        
        datamat = scio.loadmat(f'{input_dir}/s{s}/S{s}_E{exercise}_A1.mat')
        raw_emg, raw_label, repetition = datamat['emg'], datamat['restimulus'], datamat['rerepetition'] 
        for rep in repetitions:
            emg = []
            for c in classes:
                mask = ((raw_label == c) & (repetition == rep)).flatten()
                rep_emg = raw_emg[mask]
                emg.append(sliding_window(rep_emg, window_length, stride))
            label = np.repeat(classes, [x.shape[0] for x in emg]) - 1
            emg = np.concatenate(emg)
            torch.save({
                'emg': emg,
                'label': label
            }, path / f'repetition{rep}.pt', pickle_protocol=5)

def filter_labels(emg, labels, ignore_list):
    emg_temp = []
    labels_temp = []
    for i in range(len(labels)):
        if labels[i] in ignore_list:
            continue
        emg_temp.append(emg[i])
        labels_temp.append(labels[i])
    return np.array(emg_temp), labels_normal(np.array(labels_temp))

def load_emg_label(repetitions, file_fmt, subjects, ignore_list):
    x, y = [], []
    for subject in subjects:
        for rep in repetitions:
            data = torch.load(file_fmt % (subject, rep))
            rep_emg = data['emg']
            rep_label = data['label']
            for i in range(len(rep_label)):
                x.append(rep_emg[i])
                y.append(rep_label[i])
    return filter_labels(x, y, ignore_list)