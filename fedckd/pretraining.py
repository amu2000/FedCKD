"""Unlabeled two-view SimCLR initialization and classification-head reset."""

import random
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class TwoViewDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _unused_label = self.dataset[index]
        return (self.transform(image), self.transform(image))


class SimCLRProjectionHead(nn.Module):
    def __init__(self, input_dim=512, output_dim=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, features):
        return self.layers(features)


@dataclass(frozen=True)
class SimCLRPretrainResult:
    epoch_losses: list[float]


def nt_xent_loss(z1, z2, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if z1.ndim != 2 or z2.ndim != 2 or z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have the same two-dimensional shape")
    if z1.shape[0] < 2:
        raise ValueError("NT-Xent requires at least two pairs")
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    representations = torch.cat((z1, z2), dim=0)
    logits = representations @ representations.T
    logits = logits / temperature
    logits.fill_diagonal_(-torch.finfo(logits.dtype).max)
    pair_count = z1.shape[0]
    targets = torch.arange(2 * pair_count, device=logits.device) + pair_count
    targets %= 2 * pair_count
    return F.cross_entropy(logits, targets)


def make_simclr_transform(image_size=28):
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.8
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def _validate_fashion_cnn(model):
    required_attributes = ("conv1", "conv2", "fc1", "fc")
    if not all((hasattr(model, name) for name in required_attributes)):
        raise ValueError("SimCLR public pretraining currently requires FashionCNN")
    if not isinstance(model.fc, nn.Linear):
        raise ValueError("FashionCNN must expose a linear classification head")


def pretrain_fashion_cnn_simclr(
    model,
    dataset,
    device,
    epochs,
    batch_size,
    learning_rate,
    temperature,
    projection_dim,
    seed,
):
    _validate_fashion_cnn(model)
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size < 2:
        raise ValueError("batch_size must be at least two")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if projection_dim <= 0:
        raise ValueError("projection_dim must be positive")
    if len(dataset) < 2:
        raise ValueError("SimCLR pretraining requires at least two samples")
    model.to(device)
    epoch_losses = []
    with isolated_rng(seed):
        projection_head = SimCLRProjectionHead(
            input_dim=model.fc.in_features, output_dim=projection_dim
        ).to(device)
        parameters = (
            list(model.conv1.parameters())
            + list(model.conv2.parameters())
            + list(model.fc1.parameters())
            + list(projection_head.parameters())
        )
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            generator=generator,
        )
        model.train()
        projection_head.train()
        for _epoch in range(epochs):
            total_loss = 0.0
            used_samples = 0
            for view1, view2 in loader:
                if view1.shape[0] < 2:
                    continue
                view1 = view1.to(device)
                view2 = view2.to(device)
                optimizer.zero_grad(set_to_none=True)
                _logits1, features1 = model(view1, return_hidden=True)
                _logits2, features2 = model(view2, return_hidden=True)
                z1 = projection_head(features1)
                z2 = projection_head(features2)
                loss = nt_xent_loss(z1, z2, temperature)
                if not torch.isfinite(loss):
                    raise FloatingPointError("SimCLR produced a non-finite loss")
                loss.backward()
                optimizer.step()
                current_batch = view1.shape[0]
                total_loss += loss.item() * current_batch
                used_samples += current_batch
            if used_samples == 0:
                raise ValueError("SimCLR pretraining produced no usable batches")
            epoch_losses.append(total_loss / used_samples)
        model.fc.reset_parameters()
    return SimCLRPretrainResult(epoch_losses=epoch_losses)


@contextmanager
def isolated_rng(seed):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
