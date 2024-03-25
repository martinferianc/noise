from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import AugMix, Normalize

from src.data.transformations import Denormalise
from src.noises.base import Noise


class NoRotationAugmix(AugMix):
    """This class is responsible for applying AugMix without the rotation augmentation.

    It is necessary to overwrite the `_augmentation_space` method to remove the rotation augmentation.
    """

    def _augmentation_space(
        self, num_bins: int, image_size: Tuple[int, int]
    ) -> Dict[str, Tuple[torch.Tensor, bool]]:
        s = {
            # op_name: (magnitudes, signed)
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, image_size[1] / 3.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, image_size[0] / 3.0, num_bins), True),
            # "Rotate": (torch.linspace(0.0, 30.0, num_bins), True), # Removed
            "Posterize": (
                4 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(),
                False,
            ),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
        if self.all_ops:
            s.update(
                {
                    "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
                    "Color": (torch.linspace(0.0, 0.9, num_bins), True),
                    "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
                    "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
                }
            )
        return s


class InputAdditiveUniformNoise(Noise):
    """This class is responsible for adding uniform noise to the inputs of a model.

    Args:
        sigma (float): The standard deviation of the noise.
    """

    def __init__(self, sigma: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert sigma >= 0, f"sigma must be non-negative, but got {sigma}"
        self._sigma = sigma

    def _pre_forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply uniform noise to the inputs of the model.

        Args:
            inputs (torch.Tensor): The inputs to the model.
            targets (torch.Tensor): The targets for the model. They are one-hot encoded.
        """
        # Sample uniform noise from [-sigma, sigma]
        inputs = inputs + \
            torch.empty_like(inputs).uniform_(-self._sigma, self._sigma)
        return inputs, targets

    def __repr__(self) -> str:
        return f"InputAdditiveUniformNoise(sigma={self._sigma})"


class InputMultiplicativeUniformNoise(Noise):
    """This class is responsible for adding uniform noise to the inputs of a model that is multiplicative.

    The mean of the noise is 1, so the noise is centered around 1. The noise is sampled from [1-sigma, 1+sigma].

    Args:
        sigma (float): The standard deviation of the noise.
    """

    def __init__(self, sigma: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert sigma >= 0, f"sigma must be non-negative, but got {sigma}"
        self._sigma = sigma

    def _pre_forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply uniform noise to the inputs of the model.

        Args:
            inputs (torch.Tensor): The inputs to the model.
            targets (torch.Tensor): The targets for the model. They are one-hot encoded.
        """
        # Sample uniform noise from [1-sigma, 1+sigma]
        inputs = inputs * torch.empty_like(inputs).uniform_(
            1 - self._sigma, 1 + self._sigma
        )
        return inputs, targets

    def __repr__(self) -> str:
        return f"InputMultiplicativeUniformNoise(sigma={self._sigma})"


class InputAdditiveGaussianNoise(Noise):
    """This class is responsible for adding Gaussian noise to the inputs of a model.

    Args:
        sigma (float): The standard deviation of the noise.
    """

    def __init__(self, sigma: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert sigma >= 0, f"sigma must be non-negative, but got {sigma}"
        self._sigma = sigma

    def _pre_forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Gaussian noise to the inputs of the model.

        Args:
            inputs (torch.Tensor): The inputs to the model.
            targets (torch.Tensor): The targets for the model. They are one-hot encoded.
        """
        inputs = inputs + torch.randn_like(inputs) * self._sigma
        return inputs, targets

    def __repr__(self) -> str:
        return f"InputAdditiveGaussianNoise(sigma={self._sigma})"


class InputMultiplicativeGaussianNoise(Noise):
    """This class is responsible for adding Gaussian noise to the inputs of a model that is multiplicative.

    The mean of the noise is 1, so the noise is centered around 1.

    Args:
        sigma (float): The standard deviation of the noise.
    """

    def __init__(self, sigma: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert sigma >= 0, f"sigma must be non-negative, but got {sigma}"
        self._sigma = sigma

    def _pre_forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Gaussian noise to the inputs of the model.

        Args:
            inputs (torch.Tensor): The inputs to the model.
            targets (torch.Tensor): The targets for the model. They are one-hot encoded.
        """
        inputs = inputs * (1 + torch.randn_like(inputs) * self._sigma)
        return inputs, targets

    def __repr__(self) -> str:
        return f"InputMultiplicativeGaussianNoise(sigma={self._sigma})"


class InputAugMixNoise(Noise):
    """This class is responsible for adding AugMix noise to the inputs of a model.

    "AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty" https://arxiv.org/abs/1912.02781

    Args:
        mean (Tuple[float, float, float]): The mean of the normalisation transform.
        std (Tuple[float, float, float]): The standard deviation of the normalisation transform.
        severity (int): The severity of base augmentation operators. Default is 3.
        width (int): The number of augmentation chains. Default is 3.
        chain_depth (int): The depth of augmentation chains. A negative value denotes stochastic depth sampled from the interval [1, 3]. Default is -1.
        alpha (float): The hyperparameter for the probability distributions. Default is 1.0.
    """

    def __init__(
        self,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        severity: int = 3,
        width: int = 3,
        chain_depth: int = -1,
        alpha: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._normalise = Normalize(mean=mean, std=std)
        self._denormalise = Denormalise(mean=mean, std=std)
        self._augmix = NoRotationAugmix(
            severity=severity, mixture_width=width, chain_depth=chain_depth, alpha=alpha
        )
        self._mean = mean
        self._std = std
        self._severity = severity
        self._width = width
        self._chain_depth = chain_depth
        self._alpha = alpha

    def _pre_forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply AugMix noise to the inputs of the model.

        Args:
            inputs (torch.Tensor): The inputs to the model.
            targets (torch.Tensor): The targets for the model. They are one-hot encoded.
        """
        aug = self._denormalise(inputs)
        aug = (aug * 255).to(torch.uint8).cpu()
        aug = self._augmix(aug)
        aug = (aug.to(torch.float32) / 255).to(inputs.device)
        inputs = self._normalise(aug)
        return inputs, targets

    def __repr__(self) -> str:
        return f"InputAugMixNoise(severity={self._severity}, width={self._width}, chain_depth={self._chain_depth}, alpha={self._alpha})"


class InputRandomCropHorizontalFlipNoise(Noise):
    """This class implements the random crop and horizontal flip augmentation on the data during training.

    Args:
        mean (Tuple[float, float, float]): The mean of the normalisation transform.
        std (Tuple[float, float, float]): The standard deviation of the normalisation transform.
        crop_size (int): The size of the crop.
        crop_padding (int): The padding to apply before cropping.
    """

    def __init__(
        self,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        crop_size: int = 32,
        crop_padding: int = 4,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._normalise = Normalize(mean=mean, std=std)
        self._denormalise = Denormalise(mean=mean, std=std)
        self._crop_size = crop_size
        self._crop_padding = crop_padding

    def _random_crop(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply random crop"""
        top = torch.randint(0, self._crop_padding + 1, (1,))
        left = torch.randint(0, self._crop_padding + 1, (1,))
        return TF.crop(
            inputs, top.item(), left.item(), self._crop_size, self._crop_size
        )

    def _pre_forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random crop and horizontal flip augmentation to the inputs and targets.

        Args:
            inputs (torch.Tensor): The inputs to the model.
            targets (torch.Tensor): The targets for the model. They are one-hot encoded.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The augmented inputs and targets.
        """
        aug = self._denormalise(inputs)
        aug = (aug * 255).to(torch.uint8).cpu()
        # do hflip with 0.5 probability
        if torch.rand(1) > 0.5:
            aug = TF.hflip(aug)
        aug = self._random_crop(aug)
        aug = (aug.to(torch.float32) / 255).to(inputs.device)
        inputs = self._normalise(aug)
        return inputs, targets

    def __repr__(self) -> str:
        return f"InputRandomCropHorizontalFlipNoise(crop_size={self._crop_size}, crop_padding={self._crop_padding})"


class InputODSNoise(Noise):
    """This class is responsible for adding ODS noise to the inputs of a model.

    Args:
        eta (float): The scaling factor for the noise.
        temperature (float): The temperature for the noise.
    """

    def __init__(
        self, eta: float, temperature: float, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        assert eta >= 0, f"eta must be non-negative, but got {eta}"
        assert (
            temperature > 0
        ), f"temperature must be non-negative, but got {temperature}"
        self._eta = eta
        self._temperature = temperature

    def _ods(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply ODS noise to the inputs of the model.

        Args:
            inputs (torch.Tensor): The inputs to the model.
            targets (torch.Tensor): The targets for the model. They are one-hot encoded.
        """
        torch.set_grad_enabled(True)
        self._model.eval()
        inputs = inputs.detach().clone().requires_grad_(True)
        y = self._model(inputs) / self._temperature
        y = F.softmax(y, dim=1)

        w_d = torch.empty_like(y).uniform_(-1.0, 1.0)
        y_w_d = y * w_d

        grad = torch.autograd.grad(
            torch.sum(y_w_d),
            inputs,
            retain_graph=True,
            create_graph=True,
        )[0]
        grad = (inputs.view(inputs.size(0), -1).size(1) ** 0.5) * F.normalize(
            grad.reshape(grad.size(0), -1), p=2, dim=1
        ).view(grad.size())
        eps = grad * self._eta
        inputs = inputs.detach() + eps
        self._model.train()
        return inputs

    def _pre_forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply ODS noise to the inputs of the model.

        Args:
            inputs (torch.Tensor): The inputs to the model.
            targets (torch.Tensor): The targets for the model. They are one-hot encoded.
        """
        inputs = self._ods(inputs)
        return inputs, targets

    def __repr__(self) -> str:
        return f"InputODSNoise(eta={self._eta}, temperature={self._temperature})"
