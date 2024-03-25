from typing import Any, Tuple

import torch

from src.noises.base import Noise


class TargetLabelSmoothingNoise(Noise):
    """This class is responsible for adding label smoothing noise to the targets of a model.

    Args:
        smoothing (float): The smoothing factor to use for the noise.
    """

    def __init__(self, smoothing: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert (
            0 <= smoothing <= 1
        ), f"smoothing must be between 0 and 1, but got {smoothing}"
        self._smoothing = smoothing

    def _pre_forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply label smoothing noise to the targets of the model.

        Args:
            inputs (torch.Tensor): The inputs to the model.
            targets (torch.Tensor): The targets for the model. They are one-hot encoded.
        """
        targets = targets * (1 - self._smoothing) + \
            self._smoothing / targets.size(1)
        return inputs, targets

    def __repr__(self) -> str:
        return f"TargetLabelSmoothingNoise(smoothing={self._smoothing})"
