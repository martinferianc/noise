from typing import Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

NO_NOISE_KEY = "no_noise"


class Noise:
    """This is an abstract class for using noise on the data during training.

    Args:
        model (torch.nn.Module): The model to use for the noise.
        noise_probability (float): The probability of applying the noise. Default: 0.0
    """

    def __init__(self, model: nn.Module, noise_probability: float = 0.0) -> None:
        super().__init__()
        self._model = model
        assert (
            0 <= noise_probability <= 1
        ), f"noise_probability must be between 0 and 1, but got {noise_probability}"
        self._noise_probability = noise_probability
        self._forward_noise_applied = False
        self._backward_noise_applied = False
        self._epoch_noise_applied = False

    def _apply_noise(self) -> bool:
        """Return whether to apply the noise or not."""
        return torch.rand(1).item() < self._noise_probability

    def pre_forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This method can be used to apply noise before the forward pass of the model.

        It returns the inputs and targets with the noise applied.
        """
        if self._apply_noise():
            self._forward_noise_applied = True
            inputs, targets = self._pre_forward(inputs, targets)
        return inputs, targets

    def _pre_forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This method is used to apply noise before the forward pass of the model."""
        return inputs, targets

    def post_forward(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This method can be used to apply noise after the forward pass of the model.

        It returns the outputs and targets with the noise applied.
        """
        if self._forward_noise_applied:
            outputs, targets = self._post_forward(outputs, targets)
            self._forward_noise_applied = False
        return outputs, targets

    def _post_forward(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This method is used to apply noise after the forward pass of the model."""
        return outputs, targets

    def pre_backward(self, loss: torch.Tensor, epoch: int) -> torch.Tensor:
        """This method can be used to apply noise before the backward pass of the model.

        It returns the loss with the noise applied.
        """
        if self._apply_noise():
            self._backward_noise_applied = True
            loss = self._pre_backward(loss, epoch)
        return loss

    def _pre_backward(self, loss: torch.Tensor, epoch: int) -> torch.Tensor:
        """This method is used to apply noise before the backward pass of the model."""
        return loss

    def post_backward(self, loss: torch.Tensor, epoch: int) -> torch.Tensor:
        """This method can be used to apply noise after the backward pass of the model.

        It returns the loss with the noise applied.
        """
        if self._backward_noise_applied:
            loss = self._post_backward(loss, epoch)
            self._backward_noise_applied = False
        return loss

    def _post_backward(self, loss: torch.Tensor, epoch: int) -> torch.Tensor:
        """This method is used to apply noise after the backward pass of the model."""
        return loss

    def pre_epoch(self, epoch: int) -> None:
        """This method can be used to apply noise before the epoch starts."""
        if self._apply_noise():
            self._epoch_noise_applied = True
            self._pre_epoch(epoch)

    def _pre_epoch(self, epoch: int) -> None:
        """This method is used to apply noise before the epoch starts."""
        pass

    def post_epoch(self, epoch: int) -> None:
        """This method can be used to apply noise after the epoch ends."""
        if self._epoch_noise_applied:
            self._post_epoch(epoch)
            self._epoch_noise_applied = False

    def _post_epoch(self, epoch: int) -> None:
        """This method is used to apply noise after the epoch ends."""
        pass


def do_not_add_noise(module: nn.Module) -> None:
    """This function disables the noise for a given module.

    Args:
        module (nn.Module): The module for which to disable the activation noise.
    """
    for m in module.modules():
        setattr(m, NO_NOISE_KEY, True)
        if m != module:
            do_not_add_noise(m)
            
            
class LinearNoiseLayer(nn.Linear):
    """This class is responsible for adding noise to the weights of a linear layer."""

    def __init__(self, weight_noises: nn.ModuleList, activation_noises: nn.ModuleList, weight: nn.Parameter, bias: Optional[nn.Parameter] = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._weight_noises = weight_noises
        self._activation_noises = activation_noises
        
        self.weight = weight
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias
        for noise in self._weight_noises:
            weight = noise(weight)
            bias = noise(bias)
        output = F.linear(x, weight, bias)
        for noise in self._activation_noises:
            output = noise(output)
        return output


class Conv1dNoiseLayer(nn.Conv1d):
    """This class is responsible for adding noise to the weights of a convolutional layer."""

    def __init__(self,  weight_noises: nn.ModuleList, activation_noises: nn.ModuleList, weight: nn.Parameter, bias: Optional[nn.Parameter] = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._weight_noises = weight_noises
        self._activation_noises = activation_noises
        
        self.weight = weight
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias
        for noise in self._weight_noises:
            weight = noise(weight)
            bias = noise(bias)
        output = F.conv1d(
            x,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        for noise in self._activation_noises:
            output = noise(output)
        return output


class Conv2dNoiseLayer(nn.Conv2d):
    """This class is responsible for adding noise to the weights of a convolutional layer."""

    def __init__(self,  weight_noises: nn.ModuleList, activation_noises: nn.ModuleList, weight: nn.Parameter, bias: Optional[nn.Parameter] = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._weight_noises = weight_noises
        self._activation_noises = activation_noises
        self.weight = weight
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias
        for noise in self._weight_noises:
            weight = noise(weight)
            bias = noise(bias)
        output = F.conv2d(
            x,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        for noise in self._activation_noises:
            output = noise(output)
        return output

