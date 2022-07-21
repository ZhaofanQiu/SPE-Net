import torch
import logging
import operator
from typing import Any, Callable, Dict, List, Optional, Union
import torch.utils.data as torchdata
from pytorchpoints.utils.registry import Registry
from pytorchpoints.utils.comm import get_world_size
from pytorchpoints.config import configurable
from pytorchpoints.utils.env import seed_all_rng
from .common import AspectRatioGroupedDataset, DatasetFromList, MapDataset, ToIterableDataset
from .samplers import InferenceSampler, RandomSubsetTrainingSampler, TrainingSampler
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler

DATASETS_REGISTRY = Registry("DATASET")
DATASETS_REGISTRY.__doc__ = """
Registry for dataset
"""


def build_dataset(cfg, stage: str = "train"):
    dataset = DATASETS_REGISTRY.get(cfg.DATASET.NAME)(cfg, stage)
    return dataset

def _train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    if dataset is None:
        dataset_base = build_dataset(cfg, stage="train")
        dataset = dataset_base.load_data(cfg)
        if mapper is None:
            mapper = dataset_base

    #if sampler is None:
    #    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    #    logger = logging.getLogger(__name__)
    #    logger.info("Using training sampler {}".format(sampler_name))
    #    if sampler_name == "TrainingSampler":
    #        sampler = TrainingSampler(len(dataset))
    #    elif sampler_name == "RandomSubsetTrainingSampler":
    #        sampler = RandomSubsetTrainingSampler(len(dataset), cfg.DATALOADER.RANDOM_SUBSET_RATIO)
    #    else:
    #        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,
        #"sampler": sampler,
        "mapper": mapper,
        "batch_size": cfg.DATALOADER.TRAIN_BATCH_SIZE,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "shuffle" :cfg.DATALOADER.SHUFFLE
    }


@configurable(from_config=_train_loader_from_config)
def build_points_train_loader(
    dataset,
    *,
    mapper,
    batch_size,
    num_workers=0,
    shuffle=True,
    collate_fn=None,
):
    """
    Build a dataloader for object detection with some default features.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). It can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``.
            If ``dataset`` is map-style, the default sampler is a :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
            Sampler must be None if ``dataset`` is iterable.
        total_batch_size (int): total batch size across all workers.
        num_workers (int): number of parallel data loading workers
        collate_fn: a function that determines how to do batching, same as the argument of
            `torch.utils.data.DataLoader`. Defaults to do no collation and return a list of
            data. No collation is OK for small batch size and simple data structures.
            If your batch size is large and each sample contains too many small tensors,
            it's more efficient to collate them in data loader.

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False, serialize=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)

    world_size = get_world_size()
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = RandomSampler(dataset) if shuffle else None

    return torchdata.DataLoader(
        dataset,
        sampler = sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
        worker_init_fn=worker_init_reset_seed,
        drop_last=True,
    )


def _test_loader_from_config(cfg, mapper=None, *, dataset=None):
    if dataset is None:
        dataset_base = build_dataset(cfg, stage="test")
        dataset = dataset_base.load_data(cfg)
        if mapper is None:
            mapper = dataset_base

    return {
        "dataset": dataset,
        "mapper": mapper,
        "batch_size": cfg.DATALOADER.TEST_BATCH_SIZE,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "sampler": InferenceSampler(len(dataset)),
    }


@configurable(from_config=_test_loader_from_config)
def build_points_test_loader(
    dataset: Union[List[Any], torchdata.Dataset],
    *,
    mapper: Callable[[Dict[str, Any]], Any],
    sampler: Optional[torchdata.Sampler] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
) -> torchdata.DataLoader:
    """
    Similar to `build_detection_train_loader`, with default batch size = 1,
    and sampler = :class:`InferenceSampler`. This sampler coordinates all workers
    to produce the exact set of all samples.

    Args:
        dataset: a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). They can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper: a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        sampler: a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`InferenceSampler`,
            which splits the dataset across all workers. Sampler must be None
            if `dataset` is iterable.
        batch_size: the batch size of the data loader to be created.
            Default to 1 image per worker since this is the standard when reporting
            inference time in papers.
        num_workers: number of parallel data loading workers
        collate_fn: same as the argument of `torch.utils.data.DataLoader`.
            Defaults to do no collation and return a list of data.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False, serialize=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
    )


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2 ** 31
    seed_all_rng(initial_seed + worker_id)
