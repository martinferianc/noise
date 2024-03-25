from typing import Any, Tuple

import torch

from src.noises.base import Noise


class InputTargetMixUpNoise(Noise):
    """This class implements the mixup augmentation on the data during training.

    Args:
        alpha (float): The alpha parameter for the beta distribution.
    """

    def __init__(self, alpha: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert alpha > 0, "The alpha parameter must be greater than zero."
        self._alpha = alpha

    def _mixup(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup augmentation to the inputs and targets.

        Args:
            inputs (torch.Tensor): The inputs to the model.
            targets (torch.Tensor): The targets for the model. They are one-hot encoded.
        """
        B = inputs.shape[0]
        l = (
            torch.distributions.Beta(self._alpha, self._alpha)
            .sample((B,))
            .to(inputs.device)
        )
        l_inputs = l.clone().view(-1)
        # Expand the `l_inputs` to the shape of `inputs`
        while len(l_inputs.shape) < len(inputs.shape):
            l_inputs = l_inputs.unsqueeze(-1)
        l_targets = l.clone().view(-1, 1)
        # Expand the `l_targets` to the shape of `targets`
        while len(l_targets.shape) < len(targets.shape):
            l_targets = l_targets.unsqueeze(-1)
        indices_1 = torch.randperm(B).to(inputs.device)
        indices_2 = torch.randperm(B).to(inputs.device)
        inputs = l_inputs * inputs[indices_1] + \
            (1 - l_inputs) * inputs[indices_2]
        targets = l_targets * targets[indices_1] + \
            (1 - l_targets) * targets[indices_2]
        return inputs, targets

    def _pre_forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup augmentation to the inputs and targets.

        Args:
            inputs (torch.Tensor): The inputs to the model.
            targets (torch.Tensor): The targets for the model. They are one-hot encoded.
        """
        inputs, targets = self._mixup(inputs, targets)
        return inputs, targets

    def __repr__(self) -> str:
        return f"InputTargetMixUpNoise(alpha={self._alpha})"


class InputTargetCMixUpNoise(Noise):
    """This class implements the C-MixUp augmentation on the data during training for regression:

    C-Mixup: Improving Generalization in Regression: https://arxiv.org/abs/2210.05775

    It calculates the distance between the targets and uses it to compute the sampling probabilities.

    Args:
        alpha (float): The alpha parameter for the beta distribution.
        sigma (float): The standard deviation of the Gaussian kernel when computing the distance between the targets.
    """

    def __init__(self, alpha: float, sigma: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert alpha > 0, "The alpha parameter must be greater than zero."
        assert sigma > 0, "The sigma parameter must be greater than zero."
        self._alpha = alpha
        self._sigma = sigma

    def _target_distance(self, targets: torch.Tensor) -> torch.Tensor:
        """This function computes the pairwise distance between all the targets.

        The targets are assumed to be in the shape of (batch_size, 1).
        The result is a tensor of shape (batch_size, batch_size).

        It is computed as an exponential: exp(-||t_i - t_j||^2 / (2 * sigma^2))
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        distances = torch.exp(
            -(torch.cdist(targets, targets) ** 2 / (2 * self._sigma**2))
        )  # Eq. 6
        # Normalize the distances such that they sum to one
        distances = distances / distances.sum(dim=1, keepdim=True)
        return distances

    def _sample_indices(self, distances: torch.Tensor) -> torch.Tensor:
        """This function samples the indices for the mixup.

        The indices are sampled according to the distances between the targets.
        The closer the targets, the more likely they are to be sampled.
        """
        return torch.multinomial(distances, num_samples=1, replacement=False).view(-1)

    def _mixup(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup augmentation to the inputs and targets.

        Args:
            inputs (torch.Tensor): The inputs to the model.
            targets (torch.Tensor): The targets for the model. They are one-hot encoded.
        """
        B = inputs.shape[0]
        l = (
            torch.distributions.Beta(self._alpha, self._alpha)
            .sample((B,))
            .to(inputs.device)
        )
        l_inputs = l.clone().view(-1)
        # Expand the `l_inputs` to the shape of `inputs`
        while len(l_inputs.shape) < len(inputs.shape):
            l_inputs = l_inputs.unsqueeze(-1)
        l_targets = l.clone().view(-1, 1)
        # Expand the `l_targets` to the shape of `targets`
        while len(l_targets.shape) < len(targets.shape):
            l_targets = l_targets.unsqueeze(-1)

        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        indices = self._sample_indices(self._target_distance(targets))
        inputs = l_inputs * inputs + (1 - l_inputs) * inputs[indices]
        targets = l_targets * targets + (1 - l_targets) * targets[indices]
        return inputs, targets.squeeze()

    def _pre_forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup augmentation to the inputs and targets.

        Args:
            inputs (torch.Tensor): The inputs to the model.
            targets (torch.Tensor): The targets for the model. They are one-hot encoded.
        """
        inputs, targets = self._mixup(inputs, targets)
        return inputs, targets
