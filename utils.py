import torch
import torch.nn as nn
from typing import List
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
from vae import VAE

class CustomDataset(Dataset):
    def __init__(self, dataset: List[torch.Tensor]):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]

def train_model(
    model: VAE,
    data: List[torch.Tensor],
    lr: float = 0.01,
    num_samples: int = 1,
    batch_size: int = 10,
    early_stopping: bool = False,
    max_epochs: int = 1000,
    min_es_iters: int = 300,
    ref_es_iters: int = 100,
    smooth_es_iters: int = 100,
    es_thresh: float = 1e-2,
    ):
    
    assert ref_es_iters < min_es_iters
    assert smooth_es_iters < min_es_iters

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    num_batches = len(data) // batch_size
    dataset = CustomDataset([
        torch.stack(data[i * batch_size : (i + 1) * batch_size])
        for i in range(num_batches)
    ])
    
    dataloader = DataLoader(dataset, shuffle=True)
    dataset_iterator = iter(dataset)
    
    tracker = defaultdict(list)
    iter_tqdm = tqdm(range(max_epochs), desc="epochs")
    for iter_idx in iter_tqdm:
        opt.zero_grad()
        batch_metrics = defaultdict(float)
        
        try:
            batch = next(dataset_iterator)
        except StopIteration:
            dataset_iterator = iter(dataloader)
            batch = next(dataset_iterator)

        elbo, metrics = model.elbo(batch, num_samples=num_samples)
        loss = - elbo

        batch_loss = loss.mean()
        for key, value in metrics.items():
            batch_metrics[key] += value.item()

        batch_loss.backward()
        opt.step()

        for key, value in batch_metrics.items():
            tracker[key].append(value)

        iter_tqdm.set_postfix(batch_metrics)

        # Early stopping.
        if early_stopping:
            if iter_idx > min_es_iters:
                curr_loss = -sum(tracker["elbo"][-smooth_es_iters:]) / smooth_es_iters
                ref_loss = (
                    -sum(
                        tracker["elbo"][-ref_es_iters - smooth_es_iters : -ref_es_iters]
                    )
                    / smooth_es_iters
                )
                if abs(ref_loss - curr_loss) < abs(es_thresh * ref_loss):
                    break

    return tracker