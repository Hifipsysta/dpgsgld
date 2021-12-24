
import torch
from matplotlib import image
from torch.utils.data import Dataset


class Make_data(Dataset):
    def __init__(self, data_path_list, label_idx_list):
        self.len = len(label_idx_list)
        self.data_path_list = data_path_list
        self.label_idx_list = label_idx_list
    
    def __getitem__(self, idx):
        data_ = torch.tensor(image.imread(self.data_path_list[idx]), dtype=torch.float32)
        label_ = torch.tensor(self.label_idx_list[idx], dtype=torch.float32).long()
        data_ = data_.permute(-1,0,1)
        return data_/255,label_
        
    def __len__(self):
        return self.len
