import itertools
import logging
import math
from collections import defaultdict
from typing import Optional
import torch
from torch.utils.data.sampler import Sampler

from pytorchpoints.utils import comm

logger = logging.getLogger(__name__)


class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.

    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)

    Note that this sampler does not shard based on pytorch DataLoader worker id.
    A sampler passed to pytorch DataLoader is used only with map-style dataset
    and will not be executed inside workers.
    But if this sampler is used in a way that it gets execute inside a dataloader
    worker, then extra work needs to be done to shard its outputs based on worker id.
    This is required so that workers don't produce identical data.
    :class:`ToIterableDataset` implements this logic.
    This note is true for all samplers in detectron2.
    """

    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        if not isinstance(size, int):
            raise TypeError(f"TrainingSampler(size=) expects an int. Got type {type(size)}.")
        if size <= 0:
            raise ValueError(f"TrainingSampler(size=) expects a positive int. Got {size}.")
        self._size = size
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g).tolist()
            else:
                yield from torch.arange(self._size).tolist()


class RandomSubsetTrainingSampler(TrainingSampler):
    """
    Similar to TrainingSampler, but only sample a random subset of indices.
    This is useful when you want to estimate the accuracy vs data-number curves by
      training the model with different subset_ratio.
    """

    def __init__(
        self,
        size: int,
        subset_ratio: float,
        shuffle: bool = True,
        seed_shuffle: Optional[int] = None,
        seed_subset: Optional[int] = None,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            subset_ratio (float): the ratio of subset data to sample from the underlying dataset
            shuffle (bool): whether to shuffle the indices or not
            seed_shuffle (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
            seed_subset (int): the seed to randomize the subset to be sampled.
                Must be the same across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        super().__init__(size=size, shuffle=shuffle, seed=seed_shuffle)

        assert 0.0 < subset_ratio <= 1.0
        self._size_subset = int(size * subset_ratio)
        assert self._size_subset > 0
        if seed_subset is None:
            seed_subset = comm.shared_random_seed()
        self._seed_subset = int(seed_subset)

        # randomly generate the subset indexes to be sampled from
        g = torch.Generator()
        g.manual_seed(self._seed_subset)
        indexes_randperm = torch.randperm(self._size, generator=g)
        self._indexes_subset = indexes_randperm[: self._size_subset]

        logger.info("Using RandomSubsetTrainingSampler......")
        logger.info(f"Randomly sample {self._size_subset} data from the original {self._size} data")

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)  # self._seed equals seed_shuffle from __init__()
        while True:
            if self._shuffle:
                # generate a random permutation to shuffle self._indexes_subset
                randperm = torch.randperm(self._size_subset, generator=g)
                yield from self._indexes_subset[randperm].tolist()
            else:
                yield from self._indexes_subset.tolist()


class InferenceSampler(Sampler):
    """
    Produce indices for inference across all workers.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, size: int):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        self._size = size
        assert size > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        shard_size = (self._size - 1) // self._world_size + 1
        begin = shard_size * self._rank
        end = min(shard_size * (self._rank + 1), self._size)
        self._local_indices = range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)