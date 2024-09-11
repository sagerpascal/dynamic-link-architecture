from typing import Any, Dict, List, Optional, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric import Fabric
from torch import Tensor
from torch.optim import Optimizer

from utils.meters import AverageMeter


class BaseLitModule(pl.LightningModule):
    """
    Lightning Base Module
    """

    def __init__(
            self,
            conf: Dict[str, Optional[Any]],
            fabric: Fabric,
            logging_prefixes: Optional[List[str]] = None,
    ):
        """
        Constructor.
        :param conf: Configuration dictionary.
        :param fabric: Fabric instance.
        :param logging_prefixes: Prefixes for logging.
        """
        super().__init__()
        self.conf = conf
        self.fabric = fabric
        if logging_prefixes is None:
            logging_prefixes = ["train", "val"]
        self.logging_prefixes = logging_prefixes
        self.avg_meters = {}
        self.current_epoch_ = 0

    def log_step(self,
                 logs: Dict[str, torch.Tensor],
                 prefix: Optional[str] = "",
                 ):
        """
        Log the values and the metrics.
        :param logs: Values that are already processed and can be logged directly with an average value
        meter. Must be dictionaries with "meter_key" and "value".
        :param prefix: Optional prefix for the logging.
        """
        for k, v in logs.items():
            meter_name = f"{prefix}/{k}" if prefix != "" else f"{k}"
            if meter_name not in self.avg_meters:
                self.avg_meters[meter_name] = AverageMeter()
            self.avg_meters[meter_name](v)

    def log_(self) -> Dict[str, float]:
        """
        Log the metrics.
        :return: Dictionary of logs.
        """
        logs = {'epoch': self.current_epoch_}
        for m_name, m in self.avg_meters.items():
            val = m.mean
            if isinstance(val, torch.Tensor):
                val = val.item()
            logs[m_name] = val
            m.reset()
        self.log_dict(logs)
        return logs

    def epoch_end(self) -> Dict[str, float]:
        """
        Callback at the end of an epoch (log data).
        :return: Dictionary of logs.
        """
        logs = self.log_()
        self.current_epoch_ += 1
        return logs


class Autoencoder(BaseLitModule):
    """
    Extract features from non-overlapping patches of an image using a VQ-VAE.
    """

    def __init__(self, conf: Dict[str, Optional[Any]], fabric: Fabric):
        """
        Constructor.
        :param conf: Configuration dictionary.
        :param fabric: Fabric instance.
        """
        super().__init__(conf, fabric, logging_prefixes=["train", "val"])
        self.model = self.configure_model()
        self.data_var = torch.mean(torch.Tensor(self.conf['dataset']['std'])).to(fabric.device) ** 2
        self.loss_f = nn.MSELoss()

    def preprocess_data_(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Preprocess the data batch.
        :param batch: Data batch, containing input data and labels.
        :return: Preprocessed data batch.
        """
        x, y = batch
        return x, y

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the model.
        :param x:
        :return: Loss, reconstructed image, perplexity, and encodings.
        """
        return self.model(x)

    def step(self, batch: Tensor, batch_idx: int, mode_prefix: str) -> Tensor:
        """
        Forward step: Forward pass, and logging.
        :param batch: Data batch, containing input data and labels.
        :param batch_idx: Index of the batch.
        :param mode_prefix: Prefix for the mode (train, val, test).
        :return: Loss of the training step.
        """
        x, y = self.preprocess_data_(batch)
        x_recon = self.forward(x)
        x_recon_bin = (x_recon > 0.5).float()
        loss = self.loss_f(x_recon, x)
        self.log_step(
            logs={"loss": loss, "MSE": F.mse_loss(x_recon, x), "MAE": F.l1_loss(x_recon, x),
                  "MSE_binary": F.mse_loss(x_recon_bin, x), "MAE_binary": F.l1_loss(x_recon_bin, x)},
            prefix=mode_prefix
        )

        return loss

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        Forward training step: Forward pass, and logging.
        :param batch: Data batch, containing input data and labels.
        :param batch_idx: Index of the batch.
        :return: Loss of the training step.
        """
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        Forward validation step: Forward pass, and logging.
        :param batch: Data batch, containing input data and labels.
        :param batch_idx: Index of the batch.
        :return: Loss of the validation step.
        """
        return self.step(batch, batch_idx, "val")

    def configure_model(self):
        """
        Configure the model, i.e. create an autoencoder instance.
        :return:
        """
        return nn.Sequential(
            # Encoder
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),

            # # Bottleneck
            # nn.Flatten(),
            # nn.Linear(256 * 2 * 2, 256),
            # nn.ReLU(True),
            # nn.Linear(256, 256 * 2 * 2),
            # nn.ReLU(True),
            # nn.Unflatten(1, (256, 2, 2)),

            # Decoder
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def configure_optimizers(self) -> Optimizer:
        """
        Configure (create instance) the optimizer.
        :return: A torch optimizer and scheduler.
        """
        opt_conf = self.conf['optimizer']
        return torch.optim.Adam(self.parameters(),
                                lr=opt_conf['lr'],
                                betas=(opt_conf['beta_1'], opt_conf['beta_2']),
                                weight_decay=opt_conf['weight_decay'])
