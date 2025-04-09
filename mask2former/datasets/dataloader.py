import torch
from torch.utils.data import DataLoader


def _collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets


def build_train_dataloader(dataset, batch_size=1, num_workers=8, pin_memory=True):
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataloader


def build_val_dataloader(dataset, batch_size=1, num_workers=8, pin_memory=True):
    return DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
