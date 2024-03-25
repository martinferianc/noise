from typing import Tuple

import torch
import torch.nn as nn

from src.noises.base import do_not_add_noise


class Reduction(nn.Module):
    """This is a helper class which performs the reduction operation.

    Args:
        reduction (str): The reduction operation to perform. Can be 'mean', 'max' or 'min'.
        dim (int): The dimension along which to perform the reduction.
    """

    def __init__(self, reduction: str, dim: int) -> None:
        super().__init__()
        self._reduction = reduction
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._reduction == "mean":
            return x.mean(dim=self._dim)
        elif self._reduction == "max":
            return x.max(dim=self._dim)[0]
        elif self._reduction == "min":
            return x.min(dim=self._dim)[0]
        else:
            raise ValueError(f"Unknown reduction {self._reduction}")

    def extra_repr(self) -> str:
        return super().extra_repr() + f", reduction={self._reduction}, dim={self._dim}"


class GlobalPoolingCNN(nn.Module):
    """A Global Pooling CNN model.

    The dropout has been purposely removed from the model and we add
    it manually if we explore activation noise.

    Args:
        input_size (Tuple[int, int]): The shape of the input tensor.
        output_size (int): Number of output classes.
        planes (Tuple[int]): The number of hidden units in the convolutional layers.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        output_size: int,
        planes: Tuple[int, int, int],
    ) -> None:
        super(GlobalPoolingCNN, self).__init__()
        _, input_channels = input_size
        hidden_channels = planes[0]
        self.layers = nn.ModuleList()
        for hidden_channels in planes:
            self.layers.append(
                nn.Conv1d(
                    in_channels=input_channels,
                    out_channels=hidden_channels,
                    kernel_size=5,
                    stride=1,
                    padding=0,
                )
            )
            self.layers.append(nn.ReLU())
            input_channels = hidden_channels
        self.layers.append(Reduction("max", dim=2))
        self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(nn.ReLU())
        self.output = nn.Linear(planes[-1], output_size)
        do_not_add_noise(self.output)
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # x: batchsize x step x 100 - the embedding size is 100
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        # batch x step x feature_size
        # now global max pooling layer
        # The number of channels are preserved to be 128 and max is along step
        x = self.output(x)
        return x
