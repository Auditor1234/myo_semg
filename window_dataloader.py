import torch
from torch.utils.data import Dataset, DataLoader


class CNNDataset(Dataset):
    def __init__(self, emg, label) -> None:
        super().__init__()
        self.emg = emg
        self.label = label
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        return self.emg[index], self.label[index]
