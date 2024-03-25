from typing import List, Tuple, Union
import torch.nn as nn
import numpy as np
import torch

from src.noises.base import do_not_add_noise


class FC(nn.Module):
    """This defines a simple fully connected model with ReLU activations.

    Args:
        input_size Union[List[int], Tuple[int]]: Input size.
        output_size (int): Output size.
        planes (List[int]): List of hidden layer sizes.
    """

    def __init__(
        self,
        input_size: Union[List[int], Tuple[int]],
        output_size: int,
        planes: List[int],
    ) -> None:
        super(FC, self).__init__()
        self.input_size = input_size
        if isinstance(self.input_size, (tuple, list)):
            self.input_size = np.prod(self.input_size)

        self.output_size = output_size
        self.layers = nn.ModuleList([nn.Flatten()])

        for i in range(len(planes) + 1):
            if i == 0 and i == len(planes):
                self.layers.append(
                    nn.Linear(self.input_size, self.output_size))
            elif i == 0:
                self.layers.append(nn.Linear(self.input_size, planes[0]))
                self.layers.append(nn.ReLU())
            elif i == len(planes):
                self.layers.append(nn.Linear(planes[-1], self.output_size))
            else:
                self.layers.append(nn.Linear(planes[i - 1], planes[i]))
                self.layers.append(nn.ReLU())

        # split the feature extractor and the output
        if len(planes) == 0:
            self.output = nn.Sequential(self.layers[-1])
            self.layers = nn.Sequential(*self.layers[:-1])
        else:
            self.output = nn.Sequential(*self.layers[-2:])
            self.layers = nn.Sequential(*self.layers[:-2])

        do_not_add_noise(self.output[-1])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.layers(inputs)
        return self.output(features)
