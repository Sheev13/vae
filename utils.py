import torch
import torch.nn as nn
from typing import List
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, dataset: List[torch.Tensor]):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]

def train_model(
    model: nn.Module,
    data: List[torch.Tensor],
    lr: float = 0.01,
    early_stopping: bool = False,
    max_epochs: int = 1000,
    min_es_iters: int = 1000,
    ref_es_iters: int = 300,
    smooth_es_iters: int = 50,
    num_samples: int = 1,
    batch_size: int = 10,
    ):
    
    num_batches = len(data) // batch_size
    dataset = CustomDataset([
        torch.stack(data[i * batch_size : (i + 1) * batch_size])
        for i in range(num_batches)
    ])
    
    dataloader = DataLoader(dataset, shuffle=True)
    dataset_iterator = iter(dataset)
    
    #TODO: finish implementation here