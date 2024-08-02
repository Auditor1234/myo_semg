import os
import torch
import numpy as np

def setup_seed(seed = 0):
    import torch
    import random
    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_model_weight(model, filename='res/best.pt'):
    file = os.path.basename(filename)
    filepath = filename.split(file)[0]
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    torch.save(model.state_dict(), filename)
    print('save done')


def save_results(file, result):
    filename = os.path.basename(file)
    filepath = file.split(filename)[0]
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    with open(file, 'a') as f:
        f.write(result)


def data_label_shuffle(data, labels):
    N = np.random.permutation(len(data))
    data = data[N]
    labels = labels[N]

    return data, labels