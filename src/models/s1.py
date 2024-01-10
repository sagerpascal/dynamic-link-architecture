from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import Fabric
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from torchvision import utils


class Conv2dFixedFilters(nn.Module):
    """
    Fixed 2D convolutional layer with 4 filters that detect straight lines.
    """

    def __init__(self, fabric: Fabric, use_larger_weights: Optional[bool] = False, threshold_f: Optional[str] = "None"):
        """
        Constructor.
        :param fabric: A Fabric instance.
        :param use_larger_weights: Whether to use larger weights that sum up to 3 instead of 1.
        :param threshold_f: Threshold function to use. Can be "None", "threshold" or "bernoulli".
        """
        super(Conv2dFixedFilters, self).__init__()
        self.threshold_f = threshold_f

        if use_larger_weights:
            # These kernels can sum up to 3.0 (with the proper 3 cells being active)
            self.weight = torch.tensor([
                [[[+0.0, -0.5, +0.0, -0.5, +0.0],
                  [+0.0, -0.5, +1.0, -0.5, +0.0],
                  [+0.0, -0.5, +1.0, -0.5, +0.0],
                  [+0.0, -0.5, +1.0, -0.5, +0.0],
                  [+0.0, -0.5, +0.0, -0.5, +0.0]]],
                [[[+0.0, +0.0, -0.5, -0.5, +0.0],
                  [+0.0, -0.5, +0.0, +1.0, -0.5],
                  [-0.5, +0.0, +1.0, +0.0, -0.5],
                  [-0.5, +1.0, +0.0, -0.5, +0.0],
                  [+0.0, -0.5, -0.5, +0.0, +0.0]]],
                [[[+0.0, +0.0, +0.0, +0.0, +0.0],
                  [-0.5, -0.5, -0.5, -0.5, -0.5],
                  [+0.0, +1.0, +1.0, +1.0, +0.0],
                  [-0.5, -0.5, -0.5, -0.5, -0.5],
                  [+0.0, +0.0, +0.0, +0.0, +0.0]]],
                [[[+0.0, -0.5, -0.5, +0.0, +0.0],
                  [-0.5, +1.0, +0.0, -0.5, +0.0],
                  [-0.5, +0.0, +1.0, +0.0, -0.5],
                  [+0.0, -0.5, +0.0, +1.0, -0.5],
                  [+0.0, +0.0, -0.5, -0.5, +0.0]]]
                # Filter could be further improved by setting 4x +0 in the middle to -0.5
            ], dtype=torch.float32, requires_grad=False).to(fabric.device)

            self.weight = torch.tensor([
                [[[-0.5, -0.5, +0.0, -0.5, -0.5],
                  [+0.0, -0.5, +1.0, -0.5, +0.0],
                  [-0.5, -0.5, +2.0, -0.5, -0.5],
                  [+0.0, -0.5, +1.0, -0.5, +0.0],
                  [-0.5, -0.5, +0.0, -0.5, -0.5]]],
                [[[-0.5, +0.0, -0.5, -0.5, +0.0],
                  [+0.0, -0.5, -0.5, +1.0, -0.5],
                  [-0.5, -0.5, +2.0, -0.5, -0.5],
                  [-0.5, +1.0, -0.5, -0.5, +0.0],
                  [+0.0, -0.5, -0.5, +0.0, -0.5]]],
                [[[-0.5, +0.0, -0.5, +0.0, -0.5],
                  [-0.5, -0.5, -0.5, -0.5, -0.5],
                  [+0.0, +1.0, +2.0, +1.0, +0.0],
                  [-0.5, -0.5, -0.5, -0.5, -0.5],
                  [-0.5, +0.0, -0.5, +0.0, -0.5]]],
                [[[+0.0, -0.5, -0.5, +0.0, -0.5],
                  [-0.5, +1.0, -0.5, -0.5, +0.0],
                  [-0.5, -0.5, +2.0, -0.5, -0.5],
                  [+0.0, -0.5, -0.5, +1.0, -0.5],
                  [-0.5, +0.0, -0.5, -0.5, +0.0]]]
                # Filter could be further improved by setting 4x +0 in the middle to -0.5
            ], dtype=torch.float32, requires_grad=False).to(fabric.device)

        else:
            # These kernels can sum up to 1.0 (with the proper 3 cells being active)
            self.weight = torch.tensor([[[[+0, -1, +0, -1, +0],
                                          [+0, -1, +2, -1, +0],
                                          [+0, -1, +2, -1, +0],
                                          [+0, -1, +2, -1, +0],
                                          [+0, -1, +0, -1, +0]]],
                                        [[[+0, +0, -1, -1, +0],
                                          [+0, -1, +0, +2, -1],
                                          [-1, +0, +2, +0, -1],
                                          [-1, +2, +0, -1, +0],
                                          [+0, -1, -1, +0, +0]]],
                                        [[[+0, +0, +0, +0, +0],
                                          [-1, -1, -1, -1, -1],
                                          [+0, +2, +2, +2, +0],
                                          [-1, -1, -1, -1, -1],
                                          [+0, +0, +0, +0, +0]]],
                                        [[[+0, -1, -1, +0, +0],
                                          [-1, +2, +0, -1, +0],
                                          [-1, +0, +2, +0, -1],
                                          [+0, -1, +0, +2, -1],
                                          [+0, +0, -1, -1, +0]]],
                                        # Filter could be further improved by setting 4x +0 in the middle to -1
                                        ], dtype=torch.float32, requires_grad=False).to(fabric.device)
            self.weight = self.weight / 6

    def apply_conv(self, x: Tensor) -> Tensor:
        """
        Performs a 2D convolution with the fixed filters.
        :param x: Image to perform the convolution on.
        :return: Extracted features.
        """
        x = F.conv2d(x, self.weight, padding="same")
        return x

    def apply_activation(self, a: Tensor) -> Tensor:
        """
        Apply activation function to the features.
        :param a: Features
        :return: Activated features
        """

        if self.threshold_f == "None":
            return torch.where(a > 0, a, 0.)
        elif self.threshold_f == "threshold":
            return torch.where(a > 0, 1., 0.)
        elif self.threshold_f == "bernoulli":
            return torch.bernoulli(torch.clip(a, 0, 1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a 2D convolution with the fixed filters.
        :param x: Image to perform the convolution on.
        :return: Extracted features.
        """
        if len(x.shape) == 5:
            result = []
            for idx in range(x.shape[1]):
                result.append(self.apply_conv(x[:, idx, ...]))
            a = torch.stack(result, dim=1)
        else:
            a = self.apply_conv(x).unsqueeze(1)
        return self.apply_activation(a)


class FixedFilterFeatureExtractor(pl.LightningModule):
    """
    PyTorch Lightning module that uses a CNN with a fixed filter.
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the model.
        :param x: Input image.
        :return: reconstructed features
        """
        return self.model(x)

    def configure_model(self) -> nn.Module:
        """
           Configures the model.
           :return: The model.
           """
        return Conv2dFixedFilters(self.fabric, **self.conf['feature_extractor']['s1_params'])

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
            ax.bar(x, hist, align='center')
            ax.set_xlabel(f'Bins form {min:.4f} to {max:.4f}')
            ax.set_title(title)

        def _plot_weights(fig, ax, weight, title):
            weight_img_list = [weight[i, j].unsqueeze(0) for j in range(weight.shape[1]) for i in
                               range(weight.shape[0])]
            # Order is [(0, 0), (1, 0), ..., (3, 0), (0, 1), ..., (3, 7)]
            # The columns show the output channels, the rows the input channels
            grid = utils.make_grid(weight_img_list, nrow=weight.shape[0], normalize=True, scale_each=True, pad_value=1)
            # grid = grid / 2 - 1/6  # Normalize to [-1/6, 1/3]
            im = ax.imshow(grid[:, 2:-2, 2:-2].permute(1, 2, 0), interpolation='none',
                           cmap="gray")  # , vmin=-1/6, vmax=1/3)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        files = []
        for layer, weight in [('feature extractor', self.model.weight)]:
            fig, axs = plt.subplots(1, 2, figsize=(16, 10))
            _hist_plot(axs[0], weight.detach().cpu(), f"Weight distribution ({layer})")
            _plot_weights(fig, axs[1], weight[:20, :20, ...].detach().cpu(), f"Weight matrix ({layer})")
            plt.tight_layout()

            fig_fp = self.conf['run']['plots'].get('store_path', None)

            if fig_fp is not None and fig_fp != "None":
                fp = Path(fig_fp) / f'weights_{layer}.png'
                plt.savefig(fp)
                files.append(fp)

            if show_plot:
                plt.show()

            plt.close()
        return files
