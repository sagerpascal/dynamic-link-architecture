from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import Fabric
from torch import Tensor
from torchvision import utils

from data.plot_images import plot_images
from utils.meters import AverageMeter
from utils.custom_print import print_logs

W_INIT = Literal['random', 'zeros', 'identity']

class LateralLayer(nn.Module):

    def __init__(self,
                 fabric: Fabric,
                 in_channels: int,
                 out_channels: int,
                 locality_size: Optional[int] = 2,
                 lr: Optional[float] = 0.0005,
                 neg_corr: Optional[bool] = False,
                 act_threshold: Optional[Union[Literal["bernoulli"] | float]] = "bernoulli",
                 square_factor: Optional[float] = 1.2,
                 support_factor: Optional[float] = 1.3,
                 n_alternative_cells: Optional[int] = 1,
                 ):
        """
        Lateral Layer trained with Hebbian Learning. The input and output of this layer are binary.

        This layer comprises two convolutional operations to implement Hebbian learning: a fixed, binary convolution
        that rearranges every input patch into a single column vector, followed by a 1 × 1 convolution that contains
        the actual weights. More precisely, suppose our original convolution has input size h × w × n_i (where h and w
        are the height and width of the convolutional filter, and n_i is the number of channels in the input), with
        n_o output channels. Then, we can first pass the input through a fixed convolution of input size h × w × n_i
        with hwn_i output channels, with a fixed weight vector set to 1 for the weights that links input x, y, i to
        output xyi (where x, y and i run from 1 to h, w and n_i respectively) and 0 everywhere else. This rearranges
        (and duplicates) the values of each input patch of the original convolution into single, non-overlapping column
        vectors. Afterwards we can apply the actual weights of the original convolution with a simple 1 × 1 convolution,
        which can be performed by a simple tensor product with appropriate broadcasting if necessary.

        This somewhat clunkier method does require two convolutional steps, as well as additional memory usage.
        However, it also provides finer-grained control.

        Proposed in https://arxiv.org/pdf/2107.01729.pdf

        :param fabric: Fabric instance.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param locality_size: Size of the locality, i.e. how many neurons are connected to each other. For
        example, if locality_size = 2, then each neuron is connected to 5 neurons on each side of it.
        :param lr: Learning rate.
        :param neg_corr: Whether to consider negative correlation during Hebbian updates.
        :param act_threshold: The activation threshold: Either "bernoulli" to sample from a Bernoulli distribution or
        a float value as fixed threshold
        """
        super().__init__()
        self.fabric = fabric
        self.n_alternative_cells = n_alternative_cells
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_feature_channels = self.in_channels - self.out_channels
        self.locality_size = locality_size
        self.neib_size = 2 * self.locality_size + 1
        self.kernel_size = (self.neib_size, self.neib_size)
        self.lr = lr
        self.neg_corr = neg_corr
        self.act_threshold = act_threshold
        self.mask = None
        self.square_factor = square_factor
        self.support_factor = support_factor

        try:
            self.act_threshold = float(self.act_threshold)
        except ValueError:
            pass

        assert self.act_threshold in ['bernoulli'] or isinstance(self.act_threshold, float), \
            f"act_threshold must be either 'bernoulli' or a float, but is {self.act_threshold}"

        self.W_rearrange = self._init_rearrange_weights()
        self.W_lateral = nn.Parameter(self._init_lateral_weights(), requires_grad=False)
        self.ts = 0
        self.x_lateral_norm_prev = None

    def get_weights(self):
        return self.W_lateral.reshape((self.out_channels, self.in_channels) + self.kernel_size)

    def update_ts(self, ts):
        """
        Set the current timestep (relevant for sparsity rate)
        :param ts: The current timestep
        """
        self.ts = ts

    def _init_rearrange_weights(self) -> Tensor:
        """
        Initialize the transpose weights.
        :return: Tensor of shape (in_channels * kernel_size[0] * kernel_size[1], in_channels, kernel_size[0],
        kernel_size[1])
        """
        W_r_shape = (self.in_channels * self.kernel_size[0] * self.kernel_size[1], self.in_channels) + self.kernel_size
        W_rearrange = torch.zeros(W_r_shape, device=self.fabric.device, requires_grad=False)

        for i in range(self.in_channels):
            for j in range(self.kernel_size[0]):
                for k in range(self.kernel_size[1]):
                    ijk = i * self.kernel_size[0] * self.kernel_size[1] + j * self.kernel_size[1] + k
                    W_rearrange[ijk, i, j, k] = 1

        return W_rearrange

    def rearrange_input(self, x: Tensor) -> Tensor:
        """
        Rearrange the input tensor x into a single column vector.
        If x is of shape (batch_size, in_channels, height, width), then the output is of shape (batch_size,
        in_channels * kernel_size[0] * kernel_size[1], height, width).

        If the output is of shape (C, 5, 5), then the rearrangement is as follows:
        # C = 0: all inputs of kernel weight at kernel position (0,0) and input channel 0
        # C = 1: all inputs of kernel weight at kernel position (0,1) and input channel 0
        # C = 4: all inputs of kernel weight at kernel position (0,4) and input channel 0
        # C = 5: all inputs of kernel weight at kernel position (1,0) and input channel 0
        # C = 24: all inputs of kernel weight at kernel position (4,4) and input channel 0
        # C = 25: all inputs of kernel weight at kernel position (0,0) and input channel 1

        i.e. x_rearranged[0] is a matrix of size (w,h) containing all inputs that affect the kernel weight at position
        (0,0) and input channel 0. Thus, it looks like this (where p is a padding value and i_xy is the input value at
        position (x,y)):

        p, p,        p,        p,        p, ...,        p
        p, p,        p,        p,        p, ...,        p
        p, p,     i_00,     i_01,     i_02, ..., i_0(w-2)
        p, p,     i_10,     i_11,     i_12, ..., i_1(w-2)
        p, p,     i_20,     i_21,     i_22, ..., i_2(w-2)
        ...,       ...,      ...,      ..., ...,     ...
        p, p, i_(h-2)0, i_(h-2)1, i_(h-2)2, ..., i_0(w-2)

        Note: only until h-2/w-2 because top right weight does not see more, i.e. is never convolved to the input at
        the bottom right.
        """
        x_rearranged = F.conv2d(x, self.W_rearrange, padding="same")
        return x_rearranged

    def _init_lateral_weights(self) -> Tensor:
        W_lateral = torch.zeros((self.out_channels, self.in_channels * self.kernel_size[0] * self.kernel_size[1], 1, 1),
                                device=self.fabric.device, requires_grad=False)

        if self.n_alternative_cells <= 1:  # TODO: is this if/else still necessary?
            for co in range(self.out_channels):
                for ci in range(self.in_channels):
                    if ci == co or ci + self.in_feature_channels == co:
                        cii = ci * self.kernel_size[0] * self.kernel_size[1] + self.locality_size * self.kernel_size[
                            1] + self.locality_size
                        W_lateral[co, cii, 0, 0] = 1

        else:
            for co in range(self.out_channels):
                for ci in range(self.in_channels):
                    if (ci < self.in_feature_channels and co // self.n_alternative_cells == ci) or ci == co + self.in_feature_channels:
                        cii = ci * self.kernel_size[0] * self.kernel_size[1] + self.locality_size * self.kernel_size[
                            1] + self.locality_size
                        W_lateral[co, cii, 0, 0] = 1

        return W_lateral

    def calculate_correlations(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        # assert torch.all((x == 0.) | (x == 1.)), "x not binary"
        # assert torch.all((y == 0.) | (y == 1.)), "y not binary"
        x_v = x.permute(0, 2, 3, 1).reshape(-1, 1, x.shape[1])
        y_v = y.permute(0, 2, 3, 1).reshape(-1, y.shape[1], 1)
        pos_co_activation = torch.clip(torch.matmul(y_v, x_v), 0, 1)
        assert torch.all(pos_co_activation >= 0) and torch.all(pos_co_activation <= 1), "pos_co_activation not in [0,1]"

        neg_co_activation = (torch.clip(torch.matmul(y_v, torch.where(x_v==0, 1., 0.)), 0, 1)
                             + torch.clip(torch.matmul(1 - y_v, x_v), 0, 1))
        assert torch.all(neg_co_activation >= 0.) and torch.all(
            neg_co_activation <= 1), "neg_co_activation not in [0,1]"
        assert not torch.any(
            (pos_co_activation > 0) * (neg_co_activation > 0)), "pos_co_activation and neg_co_activation overlap"
        return pos_co_activation, neg_co_activation

    def hebbian_update(self, x: Tensor, y: Tensor):
        """
        Update the weights according to the Hebbian rule.
        :param x: Input tensor of shape (batch_size, in_channels * kernel_size[0] * kernel_size[1], height, width).
        :param y: Output tensor of shape (batch_size, out_channels, height, width).
        """
        pos_co_activation, neg_co_activation = self.calculate_correlations(x, y)

        if self.neg_corr:
            update = torch.mean((pos_co_activation - neg_co_activation), dim=0)
        else:
            update = torch.mean(pos_co_activation, dim=0)

        update = torch.where(update > 0., update, 0.)
        update = (update - update.min()) / (update.max() - update.min() + 1e-10)

        self.W_lateral.data += self.lr * update.view(self.W_lateral.shape)
        self.W_lateral.data = torch.clip(self.W_lateral.data, 0., 1.)

        if self.n_alternative_cells <= 1:
            self.W_lateral.data = self.W_lateral.data / (1e-10 + .2 * torch.sqrt(
                torch.sum(self.W_lateral.data ** 2, dim=[1, 2, 3], keepdim=True)))  # Weight normalization

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Dict[str, float]]:
        with torch.no_grad():
            x_rearranged = self.rearrange_input(x)
            x_rearranged = torch.where(x_rearranged > 0.5, 1., 0.)
            assert torch.all(
                (x_rearranged == 0.) | (x_rearranged == 1.)), "x_rearranged not binary -> Torch Config Error"
            x_lateral = F.conv2d(x_rearranged, self.W_lateral, padding="same", )

            # ----------------------------------------------------------------------------------------------------
            #   FIND THE CHANNEL THAT FITS THE OUTPUT BEST
            if self.n_alternative_cells > 1 and self.ts == 0:

                assert x_lateral.shape[0] == 1, "only works with batch size = 1 atm."

                d1 = x.shape[2] * x.shape[3]
                d2 = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
                w2 = (self.W_lateral.reshape(self.out_channels, d2, 1).permute(1, 0, 2) > 0).float()
                x2 = x_rearranged.reshape(d2, 1, d1).permute(0, 1, 2)
                pos_corr = torch.matmul(w2, x2)  # 40,5324,1 * 40,1,1024
                neg_corr = torch.matmul(w2, 1 - x2) + torch.matmul(1 - w2, x2)
                pos_corr = pos_corr.permute(2, 1, 0).reshape(d1, self.in_feature_channels,
                                                               self.n_alternative_cells, d2)
                neg_corr = neg_corr.permute(2, 1, 0).reshape(d1, self.in_feature_channels,
                                                               self.n_alternative_cells, d2)

                # correlation shape: (batch_size*H*W, out_channels, alt_channels, in_channels*kernel_w*kernel_h)
                # Goal for every position in the channel 0, one of the alternative output channels should be active
                pos_neg_corr_avg = torch.mean(pos_corr - neg_corr, dim=-1)

                best_channel = torch.argmax(pos_neg_corr_avg, dim=2)
                best_channel = best_channel.reshape((x_lateral.shape[0],) + x_lateral.shape[2:] + (-1,))

                self.mask = (torch.arange(0, self.n_alternative_cells).view(1, 1, 1, 1,
                                                                            -1).cuda() == best_channel.unsqueeze(
                    -1))
                assert torch.all(torch.sum(self.mask, dim=4) == 1)
                self.mask = self.mask.permute(0, 3, 4, 1, 2)

            # ----------------------------------------------------------------------------------------------------

            # reduce weight at a certain point if it is too high (Inhibition)
            max_support = self.support_factor * self.kernel_size[0]
            x_lateral_norm = torch.where(x_lateral < max_support, x_lateral,
                                         max_support - .5 * (x_lateral - max_support))

            # Normalize by dividing through the sum of the weights
            # x_lateral_norm = x_lateral_norm / (1e-10 + torch.sum(self.W_lateral.data, dim=(1, 2, 3)).view(1, -1, 1, 1))

            if self.n_alternative_cells <= 1:  # TODO: is this if/else necessary?
                # Bring activation in range [0, 1]
                x_lateral_norm_s = x_lateral_norm.shape
                x_lateral_norm /= (
                        1e-10 + x_lateral_norm.view(-1, x_lateral_norm_s[2] * x_lateral_norm_s[3]).max(1)[0].view(
                    x_lateral_norm_s[:2] + (1, 1)))

            else:
                # Normalize per alternative channel
                x_lateral_norm_s = x_lateral_norm.shape
                x_lateral_norm = x_lateral_norm.reshape((x_lateral_norm.shape[0],
                                                         x_lateral_norm.shape[1] // self.n_alternative_cells,
                                                         self.n_alternative_cells) + x_lateral_norm.shape[2:])

                x_lateral_norm = x_lateral_norm * self.mask
                # x_lateral_norm = torch.where(x_lateral_norm < torch.max(x_lateral_norm.reshape(x_lateral_norm.shape[:2] + (-1,)), dim=2)[0].view(x_lateral_norm.shape[:2] + (1, 1, 1)) * 0.6, 0, x_lateral_norm)

                x_lateral_norm_alt_max = x_lateral_norm.view(x_lateral_norm.shape[:2] + (-1,)).max(dim=2)[0]
                x_lateral_norm = x_lateral_norm / (
                        1e-10 + x_lateral_norm_alt_max.reshape(x_lateral_norm_alt_max.shape + (1, 1, 1)))
                x_lateral_norm = x_lateral_norm.reshape(x_lateral_norm_s)

            if self.act_threshold == "bernoulli":
                # in very rare cases (e.g. once in 20 epochs), there is a rounding error and it can happen that p is not in [0,1]
                x_lateral_norm = torch.clip(x_lateral_norm, 0, 1)

                p = torch.clip(x_lateral_norm ** self.square_factor[self.ts], 0, 1)
                assert torch.all((p >= 0) & (p <= 1)), "p not in [0,1]"
                x_lateral_bin = torch.bernoulli(p)
            else:
                x_lateral_bin = (x_lateral_norm ** self.square_factor[self.ts] >= self.act_threshold).float()


            stats = {
                "S2/avg_support_active": x_lateral[x_lateral_bin > 0].mean().item(),
                "S2/std_support_active": x_lateral[x_lateral_bin > 0].std().item(),
                "S2/min_support_active": x_lateral[x_lateral_bin > 0].min().item() if torch.sum(
                    x_lateral_bin > 0) > 0 else 0,
                "S2/max_support_active": x_lateral[x_lateral_bin > 0].max().item() if torch.sum(
                    x_lateral_bin > 0) > 0 else 0,
                "S2/avg_support_inactive": x_lateral[x_lateral_bin <= 0].mean().item(),
                "S2/std_support_inactive": x_lateral[x_lateral_bin <= 0].std().item(),
                "S2/min_support_inactive": x_lateral[x_lateral_bin <= 0].min().item() if torch.sum(
                    x_lateral_bin <= 0) > 0 else 0,
                "S2/max_support_inactive": x_lateral[x_lateral_bin <= 0].max().item() if torch.sum(
                    x_lateral_bin <= 0) > 0 else 0,
                "S2/norm_factor": torch.mean(x_lateral / (1e-10 + x_lateral_norm)).item()
            }

            return x_lateral_norm, x_lateral_bin, stats

class LateralLayerEfficientNetwork1L(nn.Module):
    """
    A model with two lateral layers. The first one increases the number of channels from [channels of the feature
    extractor] to [channels of the lateral network]
    """

    def __init__(self, conf: Dict[str, Optional[Any]], fabric: Fabric):
        """
        Constructor.
        :param conf: Configuration dict.
        :param fabric: Fabric instance.
        """
        super().__init__()
        self.conf = conf
        self.fabric = fabric
        self.avg_value_meter = {}

        lm_conf = self.conf["lateral_model"]
        self.out_channels = self.conf["lateral_model"]["channels"] * conf['n_alternative_cells']
        self.in_channels = self.conf["feature_extractor"]["out_channels"] + self.out_channels

        self.s2 = LateralLayer(
            self.fabric,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_alternative_cells=conf["n_alternative_cells"],
            **lm_conf["s2_params"],
        )

    def update_ts(self, ts):
        self.s2.update_ts(ts)

    def get_layer_weights(self):
        return {"S2": self.s2.get_weights()}

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass over multiple timesteps
        :param x: The input tensor (extracted features of the image)
        :return: Extracted features thar were fed into the lateral layer (binarized version of features from feature
        extractor), Features extracted by the lateral layers (binarized), Features extracted by the lateral layers (as
        float)
        """
        with torch.no_grad():
            act, act_bin, logs = self.s2(x)
        for k, v in logs.items():
            if k not in self.avg_value_meter:
                self.avg_value_meter[k] = AverageMeter()
            self.avg_value_meter[k](v, weight=x.shape[0])
        return act, act_bin

    def get_model_weight_stats(self) -> Dict[str, float]:
        """
        Get statistics of the model weights.
        :return: Dictionary with statistics.
        """
        stats = {}
        for layer, weight in self.get_layer_weights().items():
            non_zero_mask = weight != 0
            stats_ = {
                f"S2/weight_mean_{layer}": torch.mean(weight).item(),
                f"S2/weight_std_{layer}": torch.std(weight).item(),
                f"S2/weight_mean_{layer}_(0_ignored)": (
                        torch.sum(weight * non_zero_mask) / torch.sum(non_zero_mask)).item(),
                f"S2/weight_min_{layer}": torch.min(weight).item(),
                f"S2/weight_max_{layer}": torch.max(weight).item(),
                f"S2/weight_above_0.9_{layer}": torch.sum(weight >= 0.9).item() / weight.numel(),
                f"S2/weight_below_0.1_{layer}": torch.sum(weight <= 0.1).item() / weight.numel(),
            }
            stats = stats | stats_
        return stats

    def get_and_reset_logs(self) -> Dict[str, float]:
        """
        Get logs for the current epoch.
        :return: Dictionary with logs.
        """
        weight_stats = self.get_model_weight_stats()
        metrics = {}
        for k, v in self.avg_value_meter.items():
            metrics[k] = v.mean
            v.reset()

        return weight_stats | metrics


class LateralNetwork(pl.LightningModule):
    """
    PyTorch Lightning module for a network with a single lateral layer.
    """

    def __init__(self, conf: Dict[str, Optional[Any]], fabric: Fabric):
        """
        Constructor.
        :param conf: Configuration dictionary.
        :param fabric: Fabric instance.
        """
        super().__init__()
        self.conf = conf
        self.fabric = fabric
        self.model = self.configure_model()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self.model.forward(x)

    def new_sample(self):
        self.model.new_sample()

    def update_ts(self, ts):
        self.model.update_ts(ts)

    def get_and_reset_logs(self) -> Dict[str, float]:
        return self.model.get_and_reset_logs()

    def plot_model_weights(self, show_plot: Optional[bool] = False) -> List[Path]:
        """
        Plot a histogram of the model weights.
        :param show_plot: Whether to show the plot.
        :return: List of paths to the plots.
        """

        def _hist_plot(ax, weight, title):
            bins = 20
            min, max = torch.min(weight).item(), torch.max(weight).item()
            hist = torch.histc(weight, bins=bins, min=min, max=max)
            x = np.linspace(min, max, bins)
            ax.bar(x, hist, align='center', width=0.25)
            ax.set_xlabel(f'Bins form {min:.4f} to {max:.4f}')
            ax.set_title(title)

        def _plot_weights(ax, weight, title):
            weight_img_list = [weight[i, j].unsqueeze(0) for j in range(weight.shape[1]) for i in
                               range(weight.shape[0])]
            # Order is [(0, 0), (1, 0), ..., (3, 0), (0, 1), ..., (3, 7)]
            # The columns show the output channels, the rows the input channels
            grid = utils.make_grid(weight_img_list, nrow=weight.shape[0], normalize=True, scale_each=True, pad_value=1)
            ax.imshow(grid.permute(1, 2, 0))
            ax.set_title(title)

        files = []
        for layer, weight in self.model.get_layer_weights().items():
            fig, axs = plt.subplots(1, 1, figsize=(10, 10), dpi=300)
            #_hist_plot(axs[0], weight.detach().cpu(), f"Weight distribution ({layer})")
            _plot_weights(axs, weight.detach().cpu(), f"Example Weight matrix ({layer})")
            plt.tight_layout()

            fig_fp = self.conf['run']['plots'].get('store_path', None)

            if fig_fp is not None:
                fp = Path(fig_fp) / f'weights_{layer}.png'
                plt.savefig(fp)
                files.append(fp)

            if show_plot:
                plt.show()

            plt.close()
        return files

    def plot_samples(self,
                     img: List[Tensor],
                     features: List[Tensor],
                     input_features: List[Tensor],
                     lateral_features: List[Tensor],
                     lateral_features_f: List[Tensor],
                     plot_input_features: Optional[bool] = True,
                     show_plot: Optional[bool] = False,
                     ) -> List[Path]:
        """
        Plot the features extracted from a single sample.
        :param img: A list of samples, i.e. of batches of the original images (shape is (B, V, C, H, W) where V is
        the number of augmented views).
        :param features: A list of samples, i.e. batches of the features extracted from the image (shape is (B, V, C,
        H, W) where V is the number of augmented views and C is the number of output channels from the feature
        extractor).
        :param input_features: A list of samples, i.e. binary version of the features that were actually fed into the
        lateral network (shape is (B, V, C, H, W) where V is the number of augmented views and C is the number of
        output channels from the feature extractor).
        :param lateral_features: A list of samples, i.e. batches of the features extracted from the lateral network
        over time (shape is (B, V, T, C, H, W) where V is the number of augmented views, T is the number of
        timesteps, and C is the number of output channels from the lateral layer).
        lateral_features_f: Same as lateral_features but the intermediate float value before binarization.
        :param plot_input_features: Whether to plot the input features or not.
        :param show_plot: Whether to show the plot or not.
        :return: List of paths to the generated plots.
        """

        def _plot_input_features(img,
                                 features,
                                 input_features,
                                 fig_fp: Optional[str] = None,
                                 show_plot: Optional[bool] = False):
            plt_images, plt_titles = [], []
            for view_idx in range(img.shape[0]):
                plt_images.append(img[view_idx])
                plt_titles.append(f"Input view {view_idx}")
                for feature_idx in range(features.shape[1]):
                    plt_images.append(features[view_idx, feature_idx])
                    plt_titles.append(f"Features V={view_idx} C={feature_idx}")
                    plt_images.append(input_features[view_idx, feature_idx])
                    plt_titles.append(f"Lateral Input V={view_idx} C={feature_idx}")
            plt_images = self._normalize_image_list(plt_images)
            plot_images(images=plt_images, titles=plt_titles, max_cols=2 * features.shape[1] + 1, plot_colorbar=True,
                        vmin=0, vmax=1, fig_fp=fig_fp, show_plot=show_plot)

        def _plot_lateral_activation_map(lateral_features,
                                         fig_fp: Optional[str] = None,
                                         show_plot: Optional[bool] = False):
            max_views = 3
            plt_images, plt_titles = [], []
            for view_idx in range(min(max_views, lateral_features.shape[0])):
                for time_idx in range(lateral_features.shape[1]):
                    for feature_idx in range(lateral_features.shape[2]):
                        plt_images.append(lateral_features[view_idx, time_idx, feature_idx])
                        plt_titles.append(
                            f"Lat. L={1 if time_idx == 0 else 2} V={view_idx} T={time_idx} C={feature_idx}")
            plt_images = self._normalize_image_list(plt_images)
            plot_images(images=plt_images, titles=plt_titles, max_cols=lateral_features.shape[2], plot_colorbar=True,
                        vmin=0, vmax=1, fig_fp=fig_fp, show_plot=show_plot)

        def _plot_lateral_heat_map(lateral_features_f, fig_fp: Optional[str] = None, show_plot: Optional[bool] = False):
            max_views = min(3, lateral_features_f.shape[0])
            v_min, v_max = lateral_features_f[:max_views].min(), lateral_features_f[:max_views].max()
            plt_images, plt_titles = [], []
            for view_idx in range(max_views):
                for time_idx in range(lateral_features_f.shape[1]):
                    for feature_idx in range(lateral_features_f.shape[2]):
                        plt_images.append(lateral_features_f[view_idx, time_idx, feature_idx])
                        plt_titles.append(
                            f"L={1 if time_idx == 0 else 2} V={view_idx} T={time_idx} C={feature_idx}")
            plot_images(images=plt_images, titles=plt_titles, max_cols=lateral_features_f.shape[2], plot_colorbar=True,
                        fig_fp=fig_fp, cmap='hot', interpolation='nearest', vmin=v_min, vmax=v_max, show_plot=show_plot)

        def _plot_lateral_output(img,
                                 features,
                                 lateral_features,
                                 fig_fp: Optional[str] = None,
                                 show_plot: Optional[bool] = False):
            max_views = 10
            plt_images, plt_titles, plt_masks = [], [], []
            for view_idx in range(min(max_views, lateral_features.shape[0])):
                plt_images.extend([img[view_idx], img[view_idx], img[view_idx]])
                plt_titles.extend([f"Input view", f"Features", f"Lat Features avg."])
                background = torch.all((features[view_idx] == 0), dim=0)
                foreground = torch.argmax(features[view_idx], dim=0)
                calc_mask = torch.where(~background, foreground + 1, 0.)
                background2 = torch.all((lateral_features[view_idx, -1] == 0), dim=0)
                foreground2 = (torch.sum((lateral_features[view_idx, -1].reshape(4, -1, 32, 32) > 0), dim=1) > 0)
                foreground2 = torch.argmax(foreground2.float(), dim=0)
                calc_mask2 = torch.where(~background2, foreground2 + 1, 0.)
                plt_masks.extend([None, calc_mask, calc_mask2])

                for t in range(lateral_features.shape[1] - 1):
                    plt_images.append(img[view_idx])
                    plt_titles.append(f"Lat. Features T={t}")
                    background = torch.all((lateral_features[view_idx, t] == 0), dim=0)
                    foreground = (torch.sum((lateral_features[view_idx, t].reshape(4, -1, 32, 32) > 0), dim=1) > 0)
                    foreground = torch.argmax(foreground.float(), dim=0)
                    calc_mask = torch.where(~background, foreground + 1, 0.)
                    plt_masks.append(calc_mask)

            plt_images = self._normalize_image_list(plt_images)
            plot_images(images=plt_images, titles=plt_titles, masks=plt_masks, max_cols=3, plot_colorbar=False,
                        vmin=0, vmax=1, mask_vmin=0, mask_vmax=4 + 1, fig_fp=fig_fp,
                        show_plot=show_plot)

        fig_fp = self.conf['run']['plots'].get('store_path', None)
        files = []
        for i, (img_i, features_i, input_features_i, lateral_features_i, lateral_features_f_i) in enumerate(
                zip(img, features, input_features, lateral_features, lateral_features_f)):
            for batch_idx in range(img_i.shape[0]):
                if fig_fp is not None:
                    fig_fp = Path(fig_fp)
                    base_name = f"sample_{i}_batch_idx_{batch_idx}"
                    if_fp = fig_fp / f'{base_name}_input_features.png'
                    am_fp = fig_fp / f'{base_name}_lateral_act_maps.png'
                    hm_fp = fig_fp / f'{base_name}_lateral_heat_maps.png'
                    lo_fp = fig_fp / f'{base_name}_lateral_output.png'
                    # files.extend([if_fp, am_fp, hm_fp, lo_fp])
                    files.extend([lo_fp])
                else:
                    if_fp, am_fp, hm_fp, lo_fp = None, None, None, None
                # if plot_input_features:
                #     _plot_input_features(img_i[batch_idx], features_i[batch_idx], input_features_i[batch_idx],
                #                          fig_fp=if_fp, show_plot=show_plot)
                # elif if_fp is not None:
                #     files.remove(if_fp)
                # _plot_lateral_activation_map(lateral_features_i[batch_idx],
                #                              fig_fp=am_fp, show_plot=show_plot)
                # _plot_lateral_heat_map(lateral_features_f_i[batch_idx],
                #                        fig_fp=hm_fp, show_plot=show_plot)
                _plot_lateral_output(img_i[batch_idx], input_features_i[batch_idx], lateral_features_i[batch_idx],
                                     fig_fp=lo_fp, show_plot=show_plot)

        return files

    def configure_model(self) -> nn.Module:
        """
        Create the model.
        :return: Model with lateral connections.
        """
        return LateralLayerEfficientNetwork1L(self.conf, self.fabric)

    def on_epoch_end(self):
        logs = self.get_and_reset_logs()
        self.log_dict(logs)
        print_logs(logs)

    def _normalize_image_list(self, img_list):
        img_list = torch.stack([i.squeeze() for i in img_list])
        img_list = (img_list - img_list.min()) / (img_list.max() - img_list.min() + 1e-9)
        img_list = [img_list[i] for i in range(img_list.shape[0])]
        return img_list