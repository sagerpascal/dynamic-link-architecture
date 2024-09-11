import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import pytz
import torch
import wandb
from lightning import LightningModule
from lightning.fabric import Fabric
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.loader import loaders_from_config
from main_training import setup_feature_extractor
from models.autoencoder_baseline import Autoencoder, BaseLitModule
from utils.config import get_config
from utils.loggers import loggers_from_conf
from utils.store_load_run import load_run, save_run

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args(parser: Optional[argparse.ArgumentParser] = None):
    """
    Parse arguments from command line.
    :param parser: Optional ArgumentParser instance.
    :return: Parsed arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Autoencoder Baseline")
    parser.add_argument("config",
                        type=str,
                        help="Path to the config file",
                        )
    parser.add_argument('--wandb',
                        action='store_true',
                        default=False,
                        dest='logging:wandb:active',
                        help='Log to wandb'
                        )
    parser.add_argument('--store',
                        type=str,
                        dest='run:store_state_path',
                        help='Path where the model will be stored'
                        )
    parser.add_argument('--load',
                        type=str,
                        dest='run:load_state_path',
                        help='Path from where the model will be loaded'
                        )

    args = parser.parse_args()
    return args

def setup_fabric(config: Dict[str, Optional[Any]] = None) -> Fabric:
    """
    Setup the Fabric instance.
    :param config: Configuration dict
    :return: Fabric instance
    """
    callbacks = []
    loggers = loggers_from_conf(config)
    fabric = Fabric(accelerator="auto", devices=1, loggers=loggers, callbacks=callbacks)
    fabric.launch()
    fabric.seed_everything(1)
    return fabric


def setup_autoencoder(config: Dict[str, Optional[Any]], fabric: Fabric) -> (BaseLitModule, Optimizer):
    """
    Setup components for training.
    :param config: Configuration dict
    :param fabric: Fabric instance
    :return: Returns the model and the optimizer
    """
    model = Autoencoder(config, fabric)
    optimizer = model.configure_optimizers()
    model, optimizer = fabric.setup(model, optimizer)
    return model, optimizer


def setup_dataloader(config: Dict[str, Optional[Any]], fabric: Fabric) -> (DataLoader, DataLoader):
    """
    Setup the dataloaders for training and testing.
    :param config: Configuration dict
    :param fabric: Fabric instance
    :return: Returns the training dataloader
    """
    train_loader, eval_loader, _ = loaders_from_config(config)
    train_loader = fabric.setup_dataloaders(train_loader)
    eval_loader = fabric.setup_dataloaders(eval_loader)

    return train_loader, eval_loader


def configure() -> Dict[str, Optional[Any]]:
    """
    Load the config based on the given console args.
    :return:
    """
    args = parse_args()
    config = get_config(args.config, args)
    if config['run']['store_state_path'] != 'None' and Path(config['run']['store_state_path']).is_dir():
        f_name = f"s1_{datetime.now(pytz.timezone('Europe/Zurich')).strftime('%Y-%m-%d_%H-%M-%S')}.ckpt"
        config['run']['store_state_path'] = config['run']['store_state_path'] + f"/{f_name}"
    if not torch.cuda.is_available():
        warnings.warn("CUDA is not available.")
    return config


def setup_modules(config: Dict[str, Optional[Any]]) -> Tuple[Fabric, BaseLitModule, Optimizer, DataLoader, DataLoader]:
    """
    Setup the modules for training.
    :param config: Configuration dict
    :return: Returns the fabric, model, optimizer, scheduler, training dataloader and testing dataloader
    """
    fabric = setup_fabric(config)
    model, optimizer = setup_autoencoder(config, fabric)
    train_dataloader, eval_dataloader = setup_dataloader(config, fabric)
    if 'load_state_path' in config['run'] and config['run']['load_state_path'] != 'None':
        config, components = load_run(config, fabric)
        model.load_state_dict(components['model'])
        optimizer.load_state_dict(components['optimizer'])
    return fabric, model, optimizer, train_dataloader, eval_dataloader


def single_train_epoch(
        config: Dict[str, Optional[Any]],
        fabric: Fabric,
        feature_extractor: LightningModule,
        model: BaseLitModule,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        epoch: int,
):
    """
    Train a single epoch.
    :param config: Configuration dict
    :param fabric: Fabric instance
    :param feature_extractor: Feature extractor
    :param model: Model to train
    :param optimizer: Optimizer to use
    :param train_dataloader: Training dataloader
    :param epoch: Current epoch
    :return: Returns the training logs
    """
    model.train()
    feature_extractor.eval() # fixed, does not require training
    for i, batch in tqdm(enumerate(train_dataloader),
                         total=len(train_dataloader),
                         desc=f"Train Epoch {epoch + 1}/{config['run']['n_epochs']}"):

        with torch.no_grad():
            batch[0] = feature_extractor(batch[0]).squeeze(1)

        optimizer.zero_grad()
        loss = model.training_step(batch, i)
        fabric.backward(loss)
        optimizer.step()


def single_eval_epoch(
        config: Dict[str, Optional[Any]],
        feature_extractor: LightningModule,
        model: BaseLitModule,
        test_dataloader: DataLoader,
        epoch: int,
):
    """
    Evaluate a single epoch.
    :param config: Configuration dict
    :param feature_extractor: Feature extractor
    :param model: The model to evaluate
    :param test_dataloader: Testing dataloader
    :param epoch: Current epoch
    :return: Returns the validation logs
    """
    model.eval()
    feature_extractor.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader),
                             total=len(test_dataloader),
                             desc=f"Validate Epoch {epoch + 1}/{config['run']['n_epochs']}"):

            batch[0] = feature_extractor(batch[0]).squeeze(1)
            model.validation_step(batch, i)


def train_model(
        config: Dict[str, Optional[Any]],
        fabric: Fabric,
        feature_extractor: LightningModule,
        model: BaseLitModule,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        eval_loader: DataLoader
):
    """
    Train the model for multiple epochs.
    :param config: Configuration dict
    :param fabric: Fabric instance
    :param feature_extractor: Feature extractor
    :param model: Model to train
    :param optimizer: Optimizer to use
    :param train_dataloader: Training dataloader
    :param eval_loader: Evaluation dataloader
    :return:
    """
    start_epoch = config['run']['current_epoch']
    best_loss = float('inf')
    for epoch in range(start_epoch, config['run']['n_epochs']):
        config['run']['current_epoch'] = epoch
        single_train_epoch(config, fabric, feature_extractor, model, optimizer, train_dataloader, epoch)
        single_eval_epoch(config, feature_extractor, model, eval_loader, epoch)
        logs = model.epoch_end()
        fabric.call("on_epoch_end", config=config, logs=logs, fabric=fabric,
                    components={"model": model, "optimizer": optimizer})

        if 'store_state_path' in config['run'] and config['run']['store_state_path'] is not None and config['run'][
            'store_state_path'] != 'None' and logs['val/loss'] < best_loss:
            save_run(config, fabric,
                     components={'feature_extractor': feature_extractor, 'lateral_network': model})
            best_loss = logs['val/loss']

    fabric.call("on_train_end")


def main():
    """
    Run the model and store the model with the lowest loss.
    """
    config = configure()
    fabric, model, optimizer, train_dataloader, eval_loader = setup_modules(config)
    if config['logging']['wandb']['active']:
        fabric.loggers[0].log_hyperparams(config)
    feature_extractor = setup_feature_extractor(config, fabric)
    train_model(config, fabric, feature_extractor, model, optimizer, train_dataloader, eval_loader)


if __name__ == '__main__':
    main()
