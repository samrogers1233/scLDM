import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import scanpy as sc
from scduo.scduo_perturbation.ae.autoencoder import *
from scduo.scduo_perturbation import configs

class OriginalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)
        return sample

class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path).values
        if self.data.ndim == 2:
            self.data = self.data[:, np.newaxis, :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Convert to tensor if not already
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)
        return sample

def create_data_loaders(RNA_data, fold_idx=1):
    # Split dataset into training, validation, and test sets
    split_ids = five_fold_split_dataset(RNA_data, seed=19193)
    train_id, val_id, test_id = split_ids[1]
    save_dataset_ids(RNA_data, train_id, val_id, test_id, fold_idx)

    RNA_data_train, RNA_data_val, RNA_data_test = RNA_data[train_id], RNA_data[val_id], RNA_data[test_id]
    
    rna_tensor_train, rna_tensor_val, rna_tensor_test = convert_to_tensors(RNA_data_train, RNA_data_val, RNA_data_test)
    
    return rna_tensor_train, rna_tensor_val, rna_tensor_test

