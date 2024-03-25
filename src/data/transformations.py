from typing import Any, Tuple, Optional, Callable
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset


class Normalise:
    """This class normalizes the data and targets to zero mean and unit variance, given the mean and standard deviation
    of the training data.

    Args:
        mean (torch.Tensor): Mean of the training data.
        std (torch.Tensor): Standard deviation of the training data.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self._mean = mean
        self._std = std

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Normalizes the data to zero mean and unit variance.

        Args:
            data (torch.Tensor): Data to normalize.
        """
        return (data - self._mean) / (self._std + 1e-8)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse normalization.

        Args:
            data (torch.Tensor): Data to inverse normalize.
        """
        return data * self._std + self._mean


class Denormalise:
    """
    Converts the input image to the original scale.

    Args:
        mean (Tuple[float, float, float]): Mean of the input image.
        std (Tuple[float, float, float]): Standard deviation of the input image.
    """

    def __init__(
        self, mean: Tuple[float, float, float], std: Tuple[float, float, float]
    ) -> None:
        self._mean = mean
        self._std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self._mean).reshape(3, 1, 1).to(x.device)
        std = torch.tensor(self._std).reshape(3, 1, 1).to(x.device)
        return x * std + mean


class RotationDataset(Dataset):
    """This is a wrapper class around another (vision) dataset that turns a classification problem to a regression problem.

    Args:
        dataset (Dataset): The dataset to wrap.
        rotation (Tuple[float, float]): The rotation range in degrees.
        transform (Callable): The transformation to apply to the data. No transformations should be applied in
                              the original dataset. Default: None.
        seed (int): Random seed. Default: None.
    """

    def __init__(
        self,
        dataset: Dataset,
        rotation: Tuple[float, float],
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._dataset = dataset
        self._rotation = rotation
        assert (
            self._rotation[0] <= self._rotation[1]
        ), f"Rotation range must be in ascending order. Got {self._rotation}."
        self._transform = transform
        self._generator = torch.Generator().manual_seed(seed)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        x, _ = self._dataset[index]
        # Rotate the image with a random angle
        angle = (
            torch.rand(1, generator=self._generator)
            * (self._rotation[1] - self._rotation[0])
            + self._rotation[0]
        )
        x = F.rotate(x, angle.item(), interpolation=InterpolationMode.NEAREST)
        # Normalize the angle to [-1, 1] with respect to the rotation range
        y = (angle - self._rotation[0]) / (
            self._rotation[1] - self._rotation[0]
        ) * 2 - 1
        # Squeeze the tensor
        y = y.squeeze()
        if self._transform is not None:
            x = self._transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self._dataset)
