import math

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm


def calculate_mean_std(dataset):
    """Calculates per channel mean and std for a dataset.

    Args:
        dataset (torch.utils.data.Dataset): the image dataset

    Returns:
        Tuple: (per channel mean, per channel std)
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    means = []
    stds = []

    for imgs, _ in tqdm(iter(loader)):
        # print(np.shape(imgs))
        N, C, H, W = np.shape(imgs)
        batch_means = np.empty(C)
        batch_stds = np.empty(C)

        for c in range(0, C):
            batch_means[c] = torch.mean(imgs[:, c, :, :])
            batch_stds[c] = torch.std(imgs[:, c, :, :])

        # print(batch_means)
        means.append(batch_means)
        stds.append(batch_stds)

    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    return mean, std
