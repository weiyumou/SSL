import math
import random

import numpy as np
import torch
from torch.distributions.poisson import Poisson
from torch.nn import functional as F
from torch.utils.data import Dataset


def calc_dn(n):
    """
    Calculate the number of derangement D_n
    :param n: The length of the sequence
    :return: A list of number for all 0 <= i <= n
    """
    ds = [1, 0]
    for i in range(2, n + 1):
        ds.append((i - 1) * (ds[i - 1] + ds[i - 2]))
    return ds


def random_derangement(n, ds):
    """
    Implementation of the algorithm for generating random derangement of length n,
    as described in the paper "Generating Random Derangements"
    retrieved at https://epubs.siam.org/doi/pdf/10.1137/1.9781611972986.7
    :param n: The length of the derangement
    :param ds: A list of lengths of derangement for all 0 <= i <= n
    :return: A random derangement
    """
    perm = list(range(n))
    mask = [False] * n

    i, u = n - 1, n - 1
    while u >= 1:
        if not mask[i]:
            j = random.randrange(i)
            while mask[j]:
                j = random.randrange(i)
            perm[i], perm[j] = perm[j], perm[i]
            p = random.random()
            if p < u * ds[u - 1] / ds[u + 1]:
                mask[j] = True
                u -= 1
            u -= 1
        i -= 1
    return perm


def k_permute(n, k, ds):
    """
    Produces a random permutation of n elements that contains a derangment of k elements
    :param n: Total number of elements
    :param k: The length of the derangement
    :param ds: A list of lengths of derangement for all 0 <= i <= n
    :return: A random permutation with a derangement of the desired length
    """
    k = min(k, n)
    indices = list(range(n))
    sel_indices = sorted(random.sample(indices, k))
    perm = random_derangement(k, ds)
    new_indices = indices.copy()
    for i, p in enumerate(perm):
        new_indices[sel_indices[i]] = indices[sel_indices[p]]
    return new_indices


class SSLTrainDataset(Dataset):
    def __init__(self, train_dataset, num_patches, num_angles, poisson_rate) -> None:
        super(SSLTrainDataset, self).__init__()
        self.train_dataset = train_dataset
        self.num_patches = num_patches
        self.num_angles = num_angles
        self.ds = calc_dn(num_patches)
        self.pdist = Poisson(rate=poisson_rate)

    def __getitem__(self, index: int):
        rotation = torch.empty(self.num_patches, dtype=torch.long).random_(self.num_angles)
        k = self.pdist.sample().int().item()
        perm = torch.tensor(k_permute(self.num_patches, k, self.ds), dtype=torch.long)
        return self.train_dataset[index][0], rotation, perm

    def __len__(self) -> int:
        return len(self.train_dataset)

    def set_poisson_rate(self, rate):
        self.pdist = Poisson(rate)


class SSLValDataset(Dataset):

    def __init__(self, val_dataset, num_patches, num_angles) -> None:
        super(SSLValDataset, self).__init__()
        self.val_dataset = val_dataset
        self.rotations = dict()
        self.perms = dict()
        ds = calc_dn(num_patches)
        for index in range(len(val_dataset)):
            self.rotations[index] = torch.empty(num_patches, dtype=torch.long).random_(num_angles)
            self.perms[index] = torch.tensor(k_permute(num_patches, num_patches, ds), dtype=torch.long)

    def __getitem__(self, index: int):
        return self.val_dataset[index][0], self.rotations[index], self.perms[index]

    def __len__(self) -> int:
        return len(self.val_dataset)


def random_rotate(images, num_patches, rotations, perms=None):
    n, c, img_h, img_w = images.size()

    patch_size = int(img_h / math.sqrt(num_patches))
    patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)
    patches = patches.reshape(n, c, patch_size, patch_size, num_patches)
    for img_idx in range(n):
        for patch_idx in range(num_patches):
            patches[img_idx, :, :, :, patch_idx] = torch.rot90(patches[img_idx, :, :, :, patch_idx],
                                                               rotations[img_idx, patch_idx].item(), [1, 2])
        if perms is not None:
            patches[img_idx] = patches[img_idx, :, :, :, perms[img_idx]]
            rotations[img_idx] = rotations[img_idx, perms[img_idx]]

    patches = patches.reshape(n, -1, num_patches)
    images = F.fold(patches, output_size=img_h, kernel_size=patch_size, stride=patch_size)
    return images, torch.flatten(rotations)


def fast_collate(batch):
    images, rotations, perms = list(zip(*batch))
    img_tensors = []
    for img in images:
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        img_tensors.append(torch.from_numpy(nump_array))

    images = torch.stack(img_tensors, dim=0).float()
    rotations = torch.stack(rotations, dim=0)
    perms = torch.stack(perms, dim=0)
    return images, rotations, perms


class DataPrefetcher():
    def __init__(self, loader, num_patches, mean, std, scale=255):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([num * scale for num in mean]).cuda().reshape(1, 3, 1, 1)
        self.std = torch.tensor([num * scale for num in std]).cuda().reshape(1, 3, 1, 1)
        self.num_patches = num_patches

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs, labels = self.preload()
        inputs.record_stream(torch.cuda.current_stream())
        labels.record_stream(torch.cuda.current_stream())
        return inputs, labels

    def preload(self):
        images, rotations, perms = next(self.loader)
        with torch.no_grad():
            inputs, labels = random_rotate(images, self.num_patches, rotations, perms)
        with torch.cuda.stream(self.stream):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            inputs = inputs.sub(self.mean).div(self.std)
        return inputs, labels
