import math

from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.distributions.poisson import Poisson
import torch
import random


class SSLTrainDataset(Dataset):
    def __init__(self, train_dataset, num_patches, num_angles) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.num_patches = num_patches
        self.num_angles = num_angles
        self.pdist = Poisson(rate=1)

    def __getitem__(self, index: int):
        rotation = torch.empty(self.num_patches, dtype=torch.long).random_(self.num_angles)
        k = self.pdist.sample().int().item()
        perm = torch.tensor(self.k_permute(self.num_patches, k), dtype=torch.long)
        return self.train_dataset[index][0], rotation, perm

    def __len__(self) -> int:
        return len(self.train_dataset)

    def set_poisson_rate(self, rate):
        self.pdist = Poisson(rate)

    @staticmethod
    def k_permute(n, k):
        k = min(k, n)
        indices = list(range(n))
        sel_indices = random.sample(indices, k)
        perm = indices.copy()
        for idx, sel_idx in enumerate(sel_indices):
            perm[sel_idx] = indices[sel_indices[(idx + 1) % k]]
        return perm


class SSLValDataset(Dataset):

    def __init__(self, val_dataset, num_patches, num_angles) -> None:
        super().__init__()
        self.val_dataset = val_dataset
        self.rotations = dict()
        self.perms = dict()
        for index in range(len(val_dataset)):
            self.rotations[index] = torch.empty(num_patches, dtype=torch.long).random_(num_angles)
            self.perms[index] = torch.randperm(num_patches, dtype=torch.long)

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
