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
    beta_nll: bool = False,
    beta: float = 0.5
    ):
    """A function that abstracts away the training loop with a single call.

    Args:
        model: the VAE instance
        data: the dataset on which to train. Must be a list of individual tensors
        lr: learning rate of Adam optimiser
        num_samples: number of samples used to estimate the expected log likelihood term in ELBO
        batch_size: batch size; number of training points propagated before applying gradient update
        early_stopping: whether to use ad hoc early stopping criterion
        max_epochs: number of training steps if early stopping criterion not satisfied
        min_es_iters: minimum epoch at which early stopping can be applied
        ref_es_iters (int, optional): _description_. Defaults to 100.
        smooth_es_iters (int, optional): _description_. Defaults to 100.
        es_thresh (float, optional): _description_. Defaults to 1e-2.
        beta_nll (bool, optional): _description_. Defaults to False.
        beta (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    
    assert ref_es_iters < min_es_iters
    assert smooth_es_iters < min_es_iters
    
    num_batches = len(data) // batch_size
    print("Constructing training batches...")
    dataset = CustomDataset([
        torch.stack(data[i * batch_size : (i + 1) * batch_size])
        for i in tqdm(range(num_batches))
    ])
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    dataloader = DataLoader(dataset, shuffle=True)
    dataset_iterator = iter(dataset)
    
    tracker = defaultdict(list)
    print("Executing training loop...")
    iter_tqdm = tqdm(range(max_epochs), desc="epochs")
    for iter_idx in iter_tqdm:
        opt.zero_grad()
        batch_metrics = defaultdict(float)
        
        try:
            batch = next(dataset_iterator)
        except StopIteration:
            dataset_iterator = iter(dataset)
            batch = next(dataset_iterator)

        elbo, metrics = model.elbo(batch, num_samples=num_samples, beta_nll=beta_nll, beta=beta)
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


def image_view(x, image_dims, colour_channels):
    return x.view(image_dims[0], image_dims[1], colour_channels)