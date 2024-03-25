from typing import Any

import torch

from src.noises.base import Noise


class ModelShrinkAndPerturbNoise(Noise):
    """This class applies the shrink and perturb noise to the model.

    At a given `epoch_frequency`, the model is shrunk and perturbed.
    The shrinking is done with respect to a `mu` parameter and a Gaussian noise with standard deviation `sigma`
    is added to the weights.

    Args:
        mu (float): The mu parameter for the shrink and perturb noise.
        sigma (float): The standard deviation of the Gaussian noise.
        epoch_frequency (int): The frequency at which to apply the noise.
    """

    def __init__(
        self, mu: float, sigma: float, epoch_frequency: int, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        assert 0 <= mu <= 1, f"mu must be between 0 and 1, but got {mu}"
        assert sigma >= 0, f"sigma must be non-negative, but got {sigma}"
        assert (
            epoch_frequency > 0
        ), f"epoch_frequency must be positive, but got {epoch_frequency}"
        self._mu = mu
        self._sigma = sigma
        self._epoch_frequency = epoch_frequency

    def _post_epoch(self, epoch: int) -> None:
        """Update the model with the shrink and perturb noise.

        Args:
            epoch (int): The current epoch.
        """
        if epoch % self._epoch_frequency == 0 and epoch > 0:
            for param in self._model.parameters():
                if param.requires_grad:
                    param.data = (
                        param.data * self._mu
                        + torch.randn_like(param.data) * self._sigma
                    )

    def __repr__(self) -> str:
        return f"ModelShrinkAndPerturbNoise(mu={self._mu}, sigma={self._sigma}, epoch_frequency={self._epoch_frequency})"
