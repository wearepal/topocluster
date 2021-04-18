from __future__ import annotations
import copy
from typing import Iterator, Optional

from torch import Tensor
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler

import pretrainedmodels
from topocluster.reduction import RandomProjector

__all__ = ["GreedyCoreSetSampler"]

class _Embedder(nn.Module):
    def __init__(self, depth: int, n_components: Optional[int] = None) -> None:
        super().__init__()
        self.net = pretrainedmodels.inceptionv4(pretrained="imagenet").features[:depth]
        self.rand_proj = RandomProjector(n_components=n_components) if n_components else None

    def __call__(self, x: Tensor) -> Tensor:
        if x.size(1) == 1:
            x = x.expand(-1, 3, -1, -1)
        embedding = self.net(x).flatten(start_dim=1)
        if self.rand_proj is not None:
            embedding = self.rand_proj.fit_transform(embedding)
        return embedding


class GreedyCoreSetSampler(Sampler[int]):
    r"""Constructs batches from 'oversampled' batches through greedy core-set approximation.

    Args:
        n_components: Number of components to randomly project the inception embeddings to.
        If set to None, no random projection will be performed and the inception embeddings will be
        used as is

    """

    def __init__(
        self,
        dataloader: DataLoader,
        num_samples: int,
        oversampling_factor: int,
        embed_depth: int,
        n_components: Optional[int] = 10,
    ):
        embedder = _Embedder(depth=embed_depth, n_components=n_components)
        embeddings = []
        dataloader = copy.copy(dataloader)
        # Shuffle needs to be set to False to ensure that the data is 'canonically' ordered
        # for construction of the lookup table
        dataloader.shuffle = False
        for batch in dataloader:
            embeddings.append(embedder(batch.x))
        embeddings = torch.cat(embeddings, dim=0).flatten(start_dim=1)
        self.dists = torch.norm(embeddings[None] - embeddings[:, None], dim=-1)
        self.num_samples = num_samples
        self.oversampling_factor = oversampling_factor
        self.num_oversampled_samples = num_samples * oversampling_factor

    def __iter__(self) -> Iterator[int]:
        # Frist sample the 'oversampled' batch from which to construc the core-set
        os_batch_idxs = torch.randperm(self.__len__())[: self.num_oversampled_samples]
        # greedy k-center core-set construction algorithm
        unsampled_idxs = torch.ones_like(os_batch_idxs)
        sampled_idxs = [int(os_batch_idxs[0])]
        unsampled_idxs[0] = 0

        while len(sampled_idxs) < self.num_samples:
            rel_idx = torch.argmax(
                torch.min(self.dists[:, sampled_idxs][:, :, unsampled_idxs], dim=1), dim=0
            )
            p = os_batch_idxs[unsampled_idxs][rel_idx]
            unsampled_idxs[p] = 0
            sampled_idxs.append(int(p))

        return iter(sampled_idxs)

    def __len__(self) -> int:
        return len(self.dists)
