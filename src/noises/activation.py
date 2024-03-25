from typing import Any, Callable, Dict, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F

from src.noises.base import Noise, NO_NOISE_KEY, LinearNoiseLayer, Conv1dNoiseLayer, Conv2dNoiseLayer


class ActivationNoiseLayer(nn.Module):
    """This class is responsible for adding noise to the activations."""

    def __init__(self) -> None:
        super().__init__()
        self._enabled = False

    def enable(self) -> None:
        """Enable the noise."""
        self._enabled = True

    def disable(self) -> None:
        """Disable the noise."""
        self._enabled = False


class ActivationAdditiveGaussianNoiseLayer(ActivationNoiseLayer):
    """This class is responsible for adding Gaussian noise to the activations.

    Args:
        sigma (float): The standard deviation of the Gaussian noise.
    """

    def __init__(self, sigma: float) -> None:
        super().__init__()
        assert sigma >= 0, f"sigma must be non-negative, but got {sigma}"
        self._sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._enabled and self.training:
            return x + torch.randn_like(x) * self._sigma
        else:
            return x

    def extra_repr(self) -> str:
        return f"sigma={self._sigma}"


class ActivationMultiplicativeGaussianNoiseLayer(ActivationNoiseLayer):
    """This class is responsible for adding Gaussian noise to the activations, but in a multiplicative way.

    The noise is sampled from a Gaussian distribution with mean 1 and standard deviation `sigma`.

    Args:
        sigma (float): The standard deviation of the Gaussian noise.
    """

    def __init__(self, sigma: float) -> None:
        super().__init__()
        assert sigma >= 0, f"sigma must be non-negative, but got {sigma}"
        self._sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._enabled and self.training:
            return x * (1 + torch.randn_like(x) * self._sigma)
        else:
            return x

    def extra_repr(self) -> str:
        return f"sigma={self._sigma}"


class ActivationAdditiveUniformNoiseLayer(ActivationNoiseLayer):
    """This class is responsible for adding Uniform noise to the activations.

    Args:
        sigma (float): The standard deviation of the Uniform noise.
    """

    def __init__(self, sigma: float) -> None:
        super().__init__()
        assert sigma >= 0, f"sigma must be non-negative, but got {sigma}"
        self._sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._enabled and self.training:
            # Sample from the uniform distribution [-sigma, sigma]
            return x + torch.empty_like(x).uniform_(-self._sigma, self._sigma)
        else:
            return x

    def extra_repr(self) -> str:
        return f"sigma={self._sigma}"


class ActivationMultiplicativeUniformNoiseLayer(ActivationNoiseLayer):
    """This class is responsible for adding Uniform noise to the activations, but in a multiplicative way.

    The noise is sampled from a Uniform distribution with mean 1 and standard deviation `sigma`.

    Args:
        sigma (float): The standard deviation of the Uniform noise.
    """

    def __init__(self, sigma: float) -> None:
        super().__init__()
        assert sigma >= 0, f"sigma must be non-negative, but got {sigma}"
        self._sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._enabled and self.training:
            # Sample from the uniform distribution [1-sigma, 1+sigma]
            return x * (1 + torch.empty_like(x).uniform_(-self._sigma, self._sigma))
        else:
            return x

    def extra_repr(self) -> str:
        return f"sigma={self._sigma}"


class ActivationDropoutNoiseLayer(ActivationNoiseLayer):
    """This class is responsible for adding Dropout noise to the activations.

    Args:
        p (float): The probability of dropout.
    """

    def __init__(self, p: float) -> None:
        super().__init__()
        assert 0 <= p <= 1, f"p must be between 0 and 1, but got {p}"
        self._p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._enabled and self.training:
            return F.dropout(x, p=self._p, training=True)
        else:
            return x

    def extra_repr(self) -> str:
        return f"p={self._p}"


class ActivationNoise(Noise):
    """This class is responsible for adding noise to the activations of a model.

    At first it replaces all the nn.Linear and nn.Conv2d layers with the corresponding noisy layers by
    changing them into nn.Sequential with the noisy layer following the nn.Linear or nn.Conv2d layer.

    Args:
        activation_kwargs (Dict[str, Any]): The keyword arguments to pass to the activation noise layer.
        activation_constructor (Callable): The constructor of the activation noise layer.
    """

    def __init__(
        self,
        activation_kwargs: Dict[str, Any],
        activation_constructor: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._activation_kwargs = activation_kwargs
        self._activation_constructor = activation_constructor
        self._hooks = []
        self._replace_layers(self._model)

    def _replace_layers(self, model: nn.Module) -> None:
        """Replace the nn.Linear, nn.Conv1d and nn.Conv2d layers with the corresponding noisy layers."""
        for name, child in model.named_children():
            if isinstance(child, (LinearNoiseLayer, Conv1dNoiseLayer, Conv2dNoiseLayer)):
                activation_noise = self._activation_constructor(**self._activation_kwargs)
                child._activation_noises.append(activation_noise)
                self._hooks.append(activation_noise)
            elif isinstance(child, (nn.Linear)) and not hasattr(child, NO_NOISE_KEY):
                activation_noise = self._activation_constructor(**self._activation_kwargs)
                setattr(
                    model,
                    name,
                    LinearNoiseLayer(
                        weight_noises = nn.ModuleList(),
                        activation_noises=nn.ModuleList([activation_noise]),
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=child.bias,
                        weight=child.weight,
                    ),
                )
                self._hooks.append(activation_noise)
            elif isinstance(child, (nn.Conv1d)) and not hasattr(
                child, NO_NOISE_KEY
            ):
                activation_noise = self._activation_constructor(**self._activation_kwargs)
                setattr(
                    model,
                    name,
                    Conv1dNoiseLayer(
                        weight_noises = nn.ModuleList(),
                        activation_noises=nn.ModuleList([activation_noise]),
                        in_channels=child.in_channels,
                        out_channels=child.out_channels,
                        kernel_size=child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=child.bias,
                        weight=child.weight,
                        padding_mode=child.padding_mode,
                    ),
                )
                self._hooks.append(activation_noise)
            elif isinstance(child, (nn.Conv2d)) and not hasattr(
                child, NO_NOISE_KEY
            ):
                activation_noise = self._activation_constructor(**self._activation_kwargs)
                setattr(
                    model,
                    name,
                    Conv2dNoiseLayer(
                        weight_noises = nn.ModuleList(),
                        activation_noises=nn.ModuleList([activation_noise]),
                        in_channels=child.in_channels,
                        out_channels=child.out_channels,
                        kernel_size=child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=child.bias,
                        padding_mode=child.padding_mode,
                        weight=child.weight,
                    ),
                )
                self._hooks.append(activation_noise)
            else:
                self._replace_layers(child)

    def _enable_noise(self) -> None:
        """Enable the noise."""
        for m in self._hooks:
            m.enable()

    def _disable_noise(self) -> None:
        """Disable the noise."""
        for m in self._hooks:
            m.disable()

    def _pre_forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply activation noise to the model.

        In this method we enable the noise, in the _post_forward method we disable it.

        Args:
            inputs (torch.Tensor): The inputs to the model.
            targets (torch.Tensor): The targets for the model. They are one-hot encoded.
        """
        self._enable_noise()
        return inputs, targets

    def _post_forward(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Disable the activation noise.

        Args:
            outputs (torch.Tensor): The outputs of the model.
            targets (torch.Tensor): The targets for the model. They are one-hot encoded.
        """
        self._disable_noise()
        return outputs, targets
