from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """A basic cross-entropy loss for classification.

    The targets are already one-hot encoded.
    """

    def _ce(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the cross-entropy loss."""
        # The 1e-8 is to prevent NaNs when the softmax output is 0.
        return torch.mean(-torch.sum(target * torch.log_softmax(output, dim=1), dim=1))

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss."""
        loss = self._ce(output, target)
        return loss

    def extra_repr(self) -> str:
        """Return a string representation of the module."""
        return "CrossEntropyLoss()"


class RegressionLoss(nn.Module):
    """A basic Negative Log Likelihood loss for regression."""

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        The first component is the mean, the second is the logvariance.
        """
        mean, logvar = output[:, 0], output[:, 1]
        var = torch.clamp(torch.exp(logvar), min=1e-4, max=1e4)
        loss = F.gaussian_nll_loss(
            mean, target, var, reduction="none").mean(dim=0)
        return loss

    def extra_repr(self) -> str:
        """Return a string representation of the module."""
        return "NegativeLogLikelihoodLoss()"


def loss_factory(task: str, **kwargs: Any) -> nn.Module:
    """Create a loss function for a given task."""
    if task == "classification":
        return ClassificationLoss(**kwargs)
    elif task == "regression":
        return RegressionLoss(**kwargs)
    else:
        raise ValueError("Unknown task {}".format(task))
