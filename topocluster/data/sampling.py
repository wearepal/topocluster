from __future__ import annotations
import copy
import math
from typing import Iterator, List, Optional

import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from torch import Tensor
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler, SequentialSampler

from kit import implements
import pretrainedmodels
from topocluster.reduction import RandomProjector
from topocluster.utils.logging import EmbeddingProgbar

__all__ = ["GreedyCoreSetSampler"]


class GreedyCoreSetSampler(Sampler[List[int]]):
    r"""Constructs batches from 'oversampled' batches through greedy core-set approximation.

    Args:
        budget: Size of the core-set (batch)
        oversampling_factor: How many times larger than the budget the batch to be sampled from
        should be.
        embed_depth: What level of the inception networks to take the features from.
        n_components: Number of components to randomly project the inception embeddings to.
        If set to None, no random projection will be performed and the inception embeddings will be
        used as is

    """
    budget: int
    _num_oversampled_samples: int

    def __init__(
        self,
        oversampling_factor: int,
        embed_depth: int,
        n_components: Optional[int] = 10,
    ):
        self.oversampling_factor = oversampling_factor
        self.embed_depth = embed_depth
        self.n_components = n_components

    def build(self, dataloader: DataLoader, trainer: Trainer) -> None:
        assert dataloader.batch_size is not None
        # Shuffle needs to be set to False to ensure that the data is 'canonically' ordered
        # for construction of the lookup table
        if not isinstance(dataloader.sampler, SequentialSampler):
            raise ValueError("dataloader must have 'shuffle=False' for embedding-generation.")
        embedder = _Embedder(depth=self.embed_depth, n_components=self.n_components)
        runner = _DatasetEmbedderRunner(embedder=embedder)
        trainer = copy.deepcopy(trainer)
        trainer.callbacks.append(EmbeddingProgbar(trainer=trainer))
        trainer.test(model=runner, test_dataloaders=dataloader, verbose=False)
        self.embeddings = runner.embeddings
        self.budget = dataloader.batch_size
        self._num_oversampled_samples = self.budget * self.oversampling_factor

    def _get_dists(self, batch_idxs: Tensor) -> Tensor:
        batch = self.embeddings[batch_idxs]
        num_images = batch.size(0)
        dist_mat = batch @ batch.t()
        sq = dist_mat.diagonal().view(num_images, 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.t()
        return dist_mat

    @implements(Sampler)
    def __iter__(self) -> Iterator[List[int]]:
        while 1:
            # Frist sample the 'oversampled' batch from which to construc the core-set
            os_batch_idxs = torch.randperm(self.budget)[: self._num_oversampled_samples]
            # Compute the euclidean distance between all pairs in said batch
            dists = self._get_dists(os_batch_idxs)
            # greedy k-center core-set construction algorithm
            unsampled_m = torch.ones_like(os_batch_idxs, dtype=torch.bool)
            sampled_idxs = [int(os_batch_idxs[0])]
            unsampled_m[0] = 0

            while len(sampled_idxs) < self.budget:
                # p := argmax min_{i\inB}(d(x, x_i)); i.e. select the sample which maximizes the
                # minimum distance (euclidean norm) to all previously selected samples
                rel_idx = torch.argmax(
                    torch.min(dists[~unsampled_m][:, unsampled_m], dim=0).values
                )
                p = [os_batch_idxs][unsampled_m][rel_idx]
                unsampled_m[unsampled_m.nonzero()[rel_idx]] = 0
                sampled_idxs.append(int(p))

            assert len(set(sampled_idxs)) == self.budget
            yield sampled_idxs
            del dists, unsampled_m, sampled_idxs

    def __len__(self) -> int:
        return int(math.inf)


class _Embedder(nn.Module):
    def __init__(self, depth: int, n_components: int | None = None) -> None:
        super().__init__()
        self.net = pretrainedmodels.inceptionv4(pretrained="imagenet").features[:depth]
        self.rand_proj = RandomProjector(n_components=n_components) if n_components else None

    def __call__(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.size(1) == 1:
            x = x.expand(-1, 3, -1, -1)
        embedding = self.net(x).flatten(start_dim=1)
        if self.rand_proj is not None:
            embedding = self.rand_proj.fit_transform(embedding)
        return embedding


class _DatasetEmbedderRunner(pl.LightningModule):
    """Wrapper for extractor model."""

    embeddings: Tensor

    def __init__(self, embedder: _Embedder):
        super().__init__()
        self.embedder = embedder

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.embedder(x)

    @implements(pl.LightningModule)
    def test_step(self, batch: Tensor | tuple[Tensor], batch_idx: int) -> Tensor:
        img = batch[0] if isinstance(batch, tuple) else batch
        return self(img)

    @implements(pl.LightningModule)
    def test_epoch_end(self, outputs: list[Tensor]) -> None:
        self.embeddings = torch.cat(outputs, dim=0)

    def get_embeddings(self) -> Tensor:
        """Get the data from the test pass."""
        return self.embeddings
