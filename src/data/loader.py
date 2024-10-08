import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from data.straight_line import StraightLine
from torch.utils.data import DataLoader, Dataset

T = TypeVar('T')
_collate_fn_t = Callable[[List[T]], Any]
T_co = TypeVar('T_co', covariant=True)
_path_t = Union[str, os.PathLike, Path]


def _get_dataset(
        dataset_config: Optional[Dict] = None,
) -> Tuple[Any, Optional[Any], Any]:
    """
    Get a dataset based on its name.
    :param dataset_config: Config of the dataset.
    :return: A dataset.
    """
    train_set = StraightLine(**dataset_config['train_dataset_params'])
    valid_set = StraightLine(**dataset_config['valid_dataset_params'])
    test_set = StraightLine(**dataset_config['test_dataset_params'])

    return train_set, valid_set, test_set


def _get_loader_safe(
        dataset: Dataset[T_co],
        batch_size: Optional[int] = 1,
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = True,
        collate_fn: Optional[_collate_fn_t] = None,
        shuffle: Optional[bool] = True,
        drop_last: Optional[bool] = False,
) -> Optional[DataLoader[T_co]]:
    """
    Get a data loader.
    :param dataset: Dataset.
    :param batch_size: Batch size.
    :param num_workers: Number of workers.
    :param pin_memory: Whether to pin memory.
    :param collate_fn: Collate function.
    :param shuffle: Whether to shuffle the dataset.
    :param drop_last: Whether to drop the last mini-batch.
    :return: Data loader.
    """
    if dataset is not None:
        train_gen = torch.Generator()
        train_gen.manual_seed(0)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last,
                          pin_memory=pin_memory, generator=train_gen, worker_init_fn=seed_worker, collate_fn=collate_fn)
    else:
        return None


def _get_torch_data_loaders(
        train_set: Optional[Dataset[T_co]] = None,
        valid_set: Optional[Dataset[T_co]] = None,
        test_set: Optional[Dataset[T_co]] = None,
        batch_size: Optional[int] = 1,
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = True,
        collate_fn: Optional[_collate_fn_t] = None,
        shuffle_train: Optional[bool] = True,
        shuffle_valid: Optional[bool] = False,
        shuffle_test: Optional[bool] = False,
        drop_last_train: Optional[bool] = False,
        drop_last_valid: Optional[bool] = False,
        drop_last_test: Optional[bool] = False,
) -> Tuple[DataLoader[T_co], Optional[DataLoader[T_co]], Optional[DataLoader[T_co]]]:
    """
    Get data loaders for train, validation and test sets.
    :param train_set: Training set.
    :param valid_set: Validation set.
    :param test_set: Test set.
    :param batch_size: Batch size.
    :param num_workers: Number of workers.
    :param pin_memory: Whether to pin memory.
    :param collate_fn: Collate function.
    :param shuffle_train: Whether to shuffle the training set.
    :param shuffle_valid: Whether to shuffle the validation set.
    :param shuffle_test: Whether to shuffle the test set.
    :param drop_last_train: Whether to drop the last mini-batch of the training set.
    :param drop_last_valid: Whether to drop the last mini-batch of the validation set.
    :param drop_last_test: Whether to drop the last mini-batch of the test set.
    :return: Data loaders for train, validation and test sets.
    """

    train_loader = _get_loader_safe(train_set, batch_size, num_workers, pin_memory, collate_fn, shuffle_train,
                                    drop_last_train)
    valid_loader = _get_loader_safe(valid_set, batch_size, num_workers, pin_memory, collate_fn, shuffle_valid,
                                    drop_last_valid)
    test_loader = _get_loader_safe(test_set, batch_size, num_workers, pin_memory, collate_fn, shuffle_test,
                                   drop_last_test)

    return train_loader, valid_loader, test_loader


def loaders_from_config(config: Dict) -> Union[Any, Any, Any]:
    """
    Get a data loader from a config.
    :param config: Config.
    :return: A data loader.
    """
    data_config = config["dataset"]
    train_set, valid_set, test_set = _get_dataset(data_config)
    return _get_torch_data_loaders(
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
        batch_size=config["run"]["batch_size"],
        num_workers=config["run"]["num_workers"],
    )
