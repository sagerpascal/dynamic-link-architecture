import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import lightning.pytorch as pl
import torch
import wandb
from lightning import Fabric
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.loader import loaders_from_config
from models.s1 import FixedFilterFeatureExtractor
from models.s2_fragments import LateralNetwork
from utils.config import get_config
from utils.custom_print import print_start, print_warn
from utils.loggers import loggers_from_conf
from utils.store_load_run import load_run, save_run


def parse_args(parser: Optional[argparse.ArgumentParser] = None):
    """
    Parse arguments from command line.
    :param parser: Optional ArgumentParser instance.
    :return: Parsed arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Lateral Connections Stage 1")
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
    parser.add_argument('--plot',
                        action='store_true',
                        default=False,
                        dest='run:plots:enable',
                        help='Plot results'
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
    # parser.add_argument('--ts',
    #                     type=int,
    #                     default=5,
    #                     dest='lateral_model:max_timesteps',
    #                     help='Number of timesteps to train the lateral model for'
    #                     )

    args = parser.parse_args()
    return args


def configure(parser: Optional[argparse.ArgumentParser] = None) -> Dict[str, Optional[Any]]:
    """
    Load the config based on the given console args.
    :return: Configuration dict.
    """
    args = parse_args(parser)
    config = get_config(args.config, args)
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        print_warn("CUDA is not available.", title="Slow training expected.")
    return config


def setup_fabric(config: Dict[str, Optional[Any]]) -> Fabric:
    """
    Setup the fabric instance.
    :param config: Configuration dict
    :return: Fabric instance.
    """
    loggers = loggers_from_conf(config)
    fabric = Fabric(accelerator="auto", devices=1, loggers=loggers, callbacks=[])
    fabric.launch()
    fabric.seed_everything(1)
    return fabric


def setup_wandb(config: Dict[str, Optional[Any]]):
    """
    Setup wandb logging.
    :param config: The configuration dict
    """
    if "wandb" in config['logging'].keys() and config['logging']['wandb']['active']:
        wandb_conf = config['logging']['wandb']
        wandb.init(project=wandb_conf['project'],
                   config=config,
                   job_type=wandb_conf['job_type'],
                   group=wandb_conf['group'])


def setup_dataloader(config: Dict[str, Optional[Any]], fabric: Fabric) -> (DataLoader, DataLoader):
    """
    Setup the dataloaders for training and testing.
    :param config: Configuration dict
    :param fabric: Fabric instance
    :return: Returns the training and evaluation dataloader
    """
    train_loader, eval_loader, _ = loaders_from_config(config)
    train_loader = fabric.setup_dataloaders(train_loader)
    eval_loader = fabric.setup_dataloaders(eval_loader)
    return train_loader, eval_loader


def setup_feature_extractor(config: Dict[str, Optional[Any]], fabric: Fabric) -> pl.LightningModule:
    """
    Setup the feature extractor model that is used to extract features from images before they are fed into the model
    leveraging lateral connections.
    :param config: Configuration dict
    :param fabric: Fabric instance
    :return: Feature extractor model.
    """
    feature_extractor = FixedFilterFeatureExtractor(config, fabric)
    feature_extractor = fabric.setup(feature_extractor)
    return feature_extractor


def cycle(
        config: Dict[str, Optional[Any]],
        feature_extractor: pl.LightningModule,
        lateral_network: LateralNetwork,
        batch: Tensor,
        store_tensors: Optional[bool] = False,
        mode: Optional[str] = "train",
):
    """
    Perform a single cycle of the model.
    :param config: Configuration dict
    :param feature_extractor: The feature extractor model to extract features from a given image.
    :param lateral_network: The network building sub-networks by using lateral connections
    :param batch: The images to process.
    :param store_tensors: Whether to store the tensors and return them.
    :param mode: The mode of the cycle, either train or eval.
    :return: The features extracted from the image, the binarized features fed into the network with lateral
    connections, the features after lateral connections (binary) and the features after lateral connections as float
    """
    assert mode in ["train", "eval"], "Mode must be either train or eval"

    with torch.no_grad():
        features = feature_extractor(batch)

    # features = feature_extractor.binarize_features(features)
    # features = torch.clip(features, 0, 1)
    # features = torch.where(features > 0., features, torch.zeros_like(features))

    features = torch.where(features > 0, 1., 0.)

    z = None

    input_features, lateral_features, lateral_features_f, l2_features, l2h_features = [], [], [], [], []
    for view_idx in range(features.shape[1]):
        x_view_features = features[:, view_idx, ...]

        if store_tensors:
            input_features.append(x_view_features)

        if z is None:
            z = torch.zeros((x_view_features.shape[0], lateral_network.model.out_channels, x_view_features.shape[2],
                             x_view_features.shape[3]), device=batch.device)

        features_lat, features_lat_float = [], []
        for t in range(config["lateral_model"]["max_timesteps"]):
            lateral_network.model.update_ts(t)
            x_in = torch.cat([x_view_features, z], dim=1)
            z_float, z = lateral_network(x_in)

            features_lat.append(z)
            if store_tensors:
                features_lat_float.append(z_float)

        features_lat = torch.stack(features_lat, dim=1)
        features_lat_median = torch.median(features_lat, dim=1)[0]
        if store_tensors:
            features_lat_float = torch.stack(features_lat_float, dim=1)

        if mode == "train":
            # Train at the end after all timesteps (use median activation per cell),
            x_rearranged = lateral_network.model.s2.rearrange_input(
                torch.cat([x_view_features, features_lat_median], dim=1))
            lateral_network.model.s2.hebbian_update(x_rearranged, features_lat_median)

        if store_tensors:
            features_lat_float_median = torch.median(features_lat_float, dim=1)[0]
            features_lat = torch.cat([features_lat, features_lat_median.unsqueeze(1)], dim=1)
            features_lat_float = torch.cat([features_lat_float, features_lat_float_median.unsqueeze(1)], dim=1)
            lateral_features.append(features_lat)
            lateral_features_f.append(features_lat_float)

    if store_tensors:
        return features, torch.stack(input_features, dim=1), torch.stack(lateral_features, dim=1), torch.stack(
            lateral_features_f, dim=1)


def single_train_epoch(
        config: Dict[str, Optional[Any]],
        feature_extractor: pl.LightningModule,
        lateral_network: LateralNetwork,
        train_loader: DataLoader,
        epoch: int,
):
    """
        Train the model for a single epoch.
        :param config: Configuration dict.
        :param feature_extractor: Feature extractor model.
        :param lateral_network: Laternal network model.
        :param train_loader: Test set dataloader.
        :param epoch: Current epoch.
        """
    feature_extractor.eval()
    lateral_network.eval()
    for i, batch in tqdm(enumerate(train_loader),
                         total=len(train_loader),
                         colour="GREEN",
                         desc=f"Train Epoch {epoch}/{config['run']['n_epochs']}"):
        cycle(config, feature_extractor, lateral_network, batch[0], store_tensors=False, mode="train")


def single_eval_epoch(
        config: Dict[str, Optional[Any]],
        feature_extractor: pl.LightningModule,
        lateral_network: LateralNetwork,
        test_loader: DataLoader,
        epoch: int,
):
    """
    Evaluate the model for a single epoch.
    :param config: Configuration dict.
    :param feature_extractor: Feature extractor model.
    :param lateral_network: Laternal network model.
    :param test_loader: Test set dataloader.
    :param epoch: Current epoch.
    """
    feature_extractor.eval()
    lateral_network.eval()
    plt_img, plt_features, plt_input_features, plt_activations, plt_activations_f = [], [], [], [], []
    for i, batch in tqdm(enumerate(test_loader),
                         total=len(test_loader),
                         colour="GREEN",
                         desc=f"Testing Epoch {epoch}/{config['run']['n_epochs']}"):
        with torch.no_grad():
            features, input_features, lateral_features, lateral_features_f = cycle(config,
                                                                                   feature_extractor,
                                                                                   lateral_network,
                                                                                   batch[0],
                                                                                   store_tensors=True,
                                                                                   mode="eval")
            plt_img.append(batch[0])
            plt_features.append(features)
            plt_input_features.append(input_features)
            plt_activations.append(lateral_features)
            plt_activations_f.append(lateral_features_f)

    plot = config['run']['plots']['enable'] and \
           (not config['run']['plots']['only_last_epoch'] or epoch == config['run']['n_epochs'])
    wandb_b = config['logging']['wandb']['active']
    store_plots = config['run']['plots'].get('store_path', False)

    assert not wandb_b or wandb_b and store_plots, "Wandb logging requires storing the plots."

    if plot:
        if epoch == 0:
            feature_extractor.plot_model_weights(show_plot=plot)
        plots_fp = lateral_network.plot_samples(plt_img,
                                                plt_features,
                                                plt_input_features,
                                                plt_activations,
                                                plt_activations_f,
                                                plot_input_features=epoch == 0,
                                                show_plot=plot)
        weights_fp = lateral_network.plot_model_weights(show_plot=plot)

        if wandb_b:
            logs = {str(pfp.name[:-4]): wandb.Image(str(pfp)) for pfp in plots_fp}
            logs |= {str(wfp.name[:-4]): wandb.Image(str(wfp)) for wfp in weights_fp}
            wandb.log(logs | {"epoch": epoch, "trainer/global_step": epoch})


def train(
        config: Dict[str, Optional[Any]],
        feature_extractor: pl.LightningModule,
        lateral_network: LateralNetwork,
        train_loader: DataLoader,
        test_loader: DataLoader,
):
    """
    Train the model.
    :param config: Configuration dict
    :param feature_extractor: Feature extractor module
    :param lateral_network: Lateral network module
    :param train_loader: Training dataloader
    :param test_loader: Testing dataloader
    """
    start_epoch = config['run']['current_epoch']

    if config['logging']['wandb']['active'] or config['run']['plots']['enable']:
        single_eval_epoch(config, feature_extractor, lateral_network, test_loader, 0)
        lateral_network.on_epoch_end()  # print logs

    for epoch in range(start_epoch, config['run']['n_epochs']):
        single_train_epoch(config, feature_extractor, lateral_network, train_loader, epoch + 1)
        single_eval_epoch(config, feature_extractor, lateral_network, test_loader, epoch + 1)
        lateral_network.on_epoch_end()
        config['run']['current_epoch'] = epoch + 1


def setup_lateral_network(config, fabric) -> LateralNetwork:
    """
    Setup the model using lateral connections.
    :param config: Configuration dict
    :param fabric: Fabric instance
    :return: Model using lateral connections.
    """
    return fabric.setup(LateralNetwork(config, fabric))


def main():
    """
    Run the model: Create modules, extract features from images and run the model leveraging lateral connections.
    """
    print_start("Starting python script 'main_lateral_connections.py'...",
                title="Training S1: Lateral Connections Toy Example")
    config = configure()
    fabric = setup_fabric(config)
    train_loader, test_loader = setup_dataloader(config, fabric)
    feature_extractor = setup_feature_extractor(config, fabric)
    lateral_network = setup_lateral_network(config, fabric)

    if 'load_state_path' in config['run'] and config['run']['load_state_path'] != 'None':
        config, state = load_run(config, fabric)
        feature_extractor.load_state_dict(state['feature_extractor'])
        lateral_network.load_state_dict(state['lateral_network'])

    feature_extractor.eval()  # does not have to be trained
    if 'store_path' in config['run']['plots'] and config['run']['plots']['store_path'] is not None and \
            config['run']['plots']['store_path'] != 'None':
        fp = Path(config['run']['plots']['store_path'])
        if not fp.exists():
            fp.mkdir(parents=True, exist_ok=True)

    setup_wandb(config)
    train(config, feature_extractor, lateral_network, train_loader, test_loader)

    if 'store_state_path' in config['run'] and config['run']['store_state_path'] is not None and config['run'][
        'store_state_path'] != 'None':
        save_run(config, fabric,
                 components={'feature_extractor': feature_extractor, 'lateral_network': lateral_network})


if __name__ == '__main__':
    main()
