import math
from typing import Any

import torch

from src.noises.base import Noise


class GradientGaussianNoise(Noise):
    """This class is responsible for adding gaussian noise to the gradients of a model.

    Args:
        eta (float): The step size of the noise.
        gamma (float): The decay rate of the noise.
    """

    def __init__(self, eta: float, gamma: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert eta > 0, f"eta must be greater than 0, but got {eta}"
        assert gamma > 0, f"gamma must be greater than 0, but got {gamma}"
        self._eta = eta
        self._gamma = gamma

    def _var(self, epoch: int) -> float:
        """Return the variance of the noise at a given epoch."""
        return self._eta / ((1 + epoch) ** self._gamma + 1e-8)

    def _post_backward(self, loss: torch.Tensor, epoch: int) -> torch.Tensor:
        """Update the gradients with respect to some model-specific logic."""
        var = self._var(epoch)
        for param in self._model.parameters():
            if param.grad is not None:
                param.grad += torch.randn_like(param.grad) * \
                    math.sqrt(var + 1e-8)
        return loss

    def __repr__(self) -> str:
        return f"GradientGaussianNoise(eta={self._eta}, gamma={self._gamma})"
