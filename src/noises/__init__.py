from typing import Any, Dict, List, Tuple

import torch.nn as nn

from src.noises.activation import (
    ActivationAdditiveGaussianNoiseLayer,
    ActivationAdditiveUniformNoiseLayer,
    ActivationDropoutNoiseLayer,
    ActivationMultiplicativeGaussianNoiseLayer,
    ActivationMultiplicativeUniformNoiseLayer,
    ActivationNoise,
)
from src.noises.base import Noise
from src.noises.gradient import GradientGaussianNoise
from src.noises.input import (
    InputAdditiveGaussianNoise,
    InputAdditiveUniformNoise,
    InputAugMixNoise,
    InputMultiplicativeGaussianNoise,
    InputMultiplicativeUniformNoise,
    InputODSNoise,
    InputRandomCropHorizontalFlipNoise,
)
from src.noises.input_target import InputTargetCMixUpNoise, InputTargetMixUpNoise
from src.noises.model import ModelShrinkAndPerturbNoise
from src.noises.target import TargetLabelSmoothingNoise
from src.noises.weight import (
    WeightAdditiveGaussianNoiseLayer,
    WeightAdditiveUniformNoiseLayer,
    WeightMultiplicativeGaussianNoiseLayer,
    WeightMultiplicativeUniformNoiseLayer,
    WeightDropconnectNoiseLayer,
    WeightNoise,
)


def noise_factory(
    model: nn.Module,
    noises: List[str],
    noise_probabilities: List[float],
    noise_kwargs: Dict[str, Any],
    data_mean: Tuple[float, ...],
    data_std: Tuple[float, ...],
    data_size: Tuple[int, ...],
) -> List[Noise]:
    """This function returns a sequence of noise objects that are applied to the model in that order.

    Args:
        model (torch.nn.Module): The model to use for the noise.
        noises (List[str]): The noise types to use.
        noise_probabilities (List[float]): The probabilities of the noise types.
        noise_kwargs (Dict[str, Any]): The keyword arguments to pass to the noise constructor.
        data_mean (Tuple[float, ...]): The mean of the data.
        data_std (Tuple[float, ...]): The standard deviation of the data.
        data_size (Tuple[int, ...]): The size of the data.
    """
    assert len(noises) == len(
        noise_probabilities
    ), f"The number of noises and noise probabilities must be the same. Got {len(noises)} noises and {len(noise_probabilities)} probabilities."
    assert len(set(noises)) == len(
        noises), f"The noises must be unique. Got {noises}."
    noise_list: List[Noise] = []

    for noise_type, noise_probability in zip(noises, noise_probabilities):
        # GRADIENT NOISE
        if noise_type == "gradient_gaussian":
            noise_list.append(
                GradientGaussianNoise(
                    model=model,
                    noise_probability=noise_probability,
                    eta=noise_kwargs.pop("noise_gradient_eta"),
                    gamma=noise_kwargs.pop("noise_gradient_gamma"),
                )
            )
        # TARGET NOISE
        elif noise_type == "target_label_smoothing":
            noise_list.append(
                TargetLabelSmoothingNoise(
                    model=model,
                    noise_probability=noise_probability,
                    smoothing=noise_kwargs.pop("noise_target_label_smoothing"),
                )
            )
        # MODEL NOISE
        elif noise_type == "model_shrink_and_perturb":
            noise_list.append(
                ModelShrinkAndPerturbNoise(
                    model=model,
                    noise_probability=noise_probability,
                    mu=noise_kwargs.pop("noise_model_shrink_and_perturb_mu"),
                    sigma=noise_kwargs.pop(
                        "noise_model_shrink_and_perturb_sigma"),
                    epoch_frequency=noise_kwargs.pop(
                        "noise_model_shrink_and_perturb_epoch_frequency"
                    ),
                )
            )
        # INPUT NOISE
        elif noise_type == "input_additive_gaussian":
            noise_list.append(
                InputAdditiveGaussianNoise(
                    model=model,
                    noise_probability=noise_probability,
                    sigma=noise_kwargs.pop(
                        "noise_input_additive_gaussian_sigma"),
                )
            )
        elif noise_type == "input_multiplicative_gaussian":
            noise_list.append(
                InputMultiplicativeGaussianNoise(
                    model=model,
                    noise_probability=noise_probability,
                    sigma=noise_kwargs.pop(
                        "noise_input_multiplicative_gaussian_sigma"),
                )
            )
        elif noise_type == "input_additive_uniform":
            noise_list.append(
                InputAdditiveUniformNoise(
                    model=model,
                    noise_probability=noise_probability,
                    sigma=noise_kwargs.pop(
                        "noise_input_additive_uniform_sigma"),
                )
            )
        elif noise_type == "input_multiplicative_uniform":
            noise_list.append(
                InputMultiplicativeUniformNoise(
                    model=model,
                    noise_probability=noise_probability,
                    sigma=noise_kwargs.pop(
                        "noise_input_multiplicative_uniform_sigma"),
                )
            )
        elif noise_type == "input_ods":
            noise_list.append(
                InputODSNoise(
                    model=model,
                    noise_probability=noise_probability,
                    eta=noise_kwargs.pop("noise_input_ods_eta"),
                    temperature=noise_kwargs.pop(
                        "noise_input_ods_temperature"),
                )
            )
        elif noise_type == "input_augmix":
            noise_list.append(
                InputAugMixNoise(
                    model=model,
                    noise_probability=noise_probability,
                    mean=data_mean,
                    std=data_std,
                    severity=noise_kwargs.pop("noise_input_augmix_severity"),
                    width=noise_kwargs.pop("noise_input_augmix_width"),
                    chain_depth=noise_kwargs.pop(
                        "noise_input_augmix_chain_depth"),
                    alpha=noise_kwargs.pop("noise_input_augmix_alpha"),
                )
            )
        elif noise_type == "input_random_crop_horizontal_flip":
            noise_list.append(
                InputRandomCropHorizontalFlipNoise(
                    model=model,
                    noise_probability=noise_probability,
                    mean=data_mean,
                    std=data_std,
                    crop_size=data_size[-1],
                    crop_padding=noise_kwargs.pop(
                        "noise_input_random_crop_horizontal_flip_crop_padding"
                    ),
                )
            )
        # INPUT TARGET NOISE
        elif noise_type == "input_target_mixup":
            noise_list.append(
                InputTargetMixUpNoise(
                    model=model,
                    noise_probability=noise_probability,
                    alpha=noise_kwargs.pop("noise_input_target_mixup_alpha"),
                )
            )
        elif noise_type == "input_target_cmixup":
            noise_list.append(
                InputTargetCMixUpNoise(
                    model=model,
                    noise_probability=noise_probability,
                    alpha=noise_kwargs.pop("noise_input_target_cmixup_alpha"),
                    sigma=noise_kwargs.pop("noise_input_target_cmixup_sigma"),
                )
            )
        # ACTIVATION NOISE
        elif noise_type == "activation_additive_gaussian":
            activation_kwargs = {
                "sigma": noise_kwargs.pop("noise_activation_additive_gaussian_sigma")
            }
            noise_list.append(
                ActivationNoise(
                    model=model,
                    noise_probability=noise_probability,
                    activation_kwargs=activation_kwargs,
                    activation_constructor=ActivationAdditiveGaussianNoiseLayer,
                )
            )
        elif noise_type == "activation_multiplicative_gaussian":
            activation_kwargs = {
                "sigma": noise_kwargs.pop(
                    "noise_activation_multiplicative_gaussian_sigma"
                )
            }
            noise_list.append(
                ActivationNoise(
                    model=model,
                    noise_probability=noise_probability,
                    activation_kwargs=activation_kwargs,
                    activation_constructor=ActivationMultiplicativeGaussianNoiseLayer,
                )
            )
        elif noise_type == "activation_additive_uniform":
            activation_kwargs = {
                "sigma": noise_kwargs.pop("noise_activation_additive_uniform_sigma")
            }
            noise_list.append(
                ActivationNoise(
                    model=model,
                    noise_probability=noise_probability,
                    activation_kwargs=activation_kwargs,
                    activation_constructor=ActivationAdditiveUniformNoiseLayer,
                )
            )
        elif noise_type == "activation_multiplicative_uniform":
            activation_kwargs = {
                "sigma": noise_kwargs.pop(
                    "noise_activation_multiplicative_uniform_sigma"
                )
            }
            noise_list.append(
                ActivationNoise(
                    model=model,
                    noise_probability=noise_probability,
                    activation_kwargs=activation_kwargs,
                    activation_constructor=ActivationMultiplicativeUniformNoiseLayer,
                )
            )
        elif noise_type == "activation_dropout":
            activation_kwargs = {"p": noise_kwargs.pop(
                "noise_activation_dropout_p")}
            noise_list.append(
                ActivationNoise(
                    model=model,
                    noise_probability=noise_probability,
                    activation_kwargs=activation_kwargs,
                    activation_constructor=ActivationDropoutNoiseLayer,
                )
            )
        # WEIGHT NOISE
        elif noise_type == "weight_additive_gaussian":
            noise_list.append(
                WeightNoise(
                    model=model,
                    noise_probability=noise_probability,
                    weight_kwargs={
                        "sigma": noise_kwargs.pop("noise_weight_additive_gaussian_sigma")
                    },
                    weight_constructor=WeightAdditiveGaussianNoiseLayer,
                )
            )
        elif noise_type == "weight_multiplicative_gaussian":
            noise_list.append(
                WeightNoise(
                    model=model,
                    noise_probability=noise_probability,
                    weight_kwargs={
                        "sigma": noise_kwargs.pop(
                            "noise_weight_multiplicative_gaussian_sigma"
                        )
                    },
                    weight_constructor=WeightMultiplicativeGaussianNoiseLayer,
                )
            )
        elif noise_type == "weight_additive_uniform":
            noise_list.append(
                WeightNoise(
                    model=model,
                    noise_probability=noise_probability,
                    weight_kwargs={
                        "sigma": noise_kwargs.pop("noise_weight_additive_uniform_sigma")
                    },
                    weight_constructor=WeightAdditiveUniformNoiseLayer,
                )
            )
        elif noise_type == "weight_multiplicative_uniform":
            noise_list.append(
                WeightNoise(
                    model=model,
                    noise_probability=noise_probability,
                    weight_kwargs={
                        "sigma": noise_kwargs.pop(
                            "noise_weight_multiplicative_uniform_sigma"
                        )
                    },
                    weight_constructor=WeightMultiplicativeUniformNoiseLayer,
                )
            )
        elif noise_type == "weight_dropconnect":
            noise_list.append(
                WeightNoise(
                    model=model,
                    noise_probability=noise_probability,
                    weight_kwargs={
                        "p": noise_kwargs.pop("noise_weight_dropconnect_p")
                    },
                    weight_constructor=WeightDropconnectNoiseLayer,
                )
            )
        elif noise_type == "none":
            noise_list.append(Noise(model=model))
        else:
            raise ValueError(f"Invalid noise type: {noise_type}")

    return noise_list


NOISE_MAPPING = {
    "gradient_gaussian": GradientGaussianNoise,
    "target_label_smoothing": TargetLabelSmoothingNoise,
    "model_shrink_and_perturb": ModelShrinkAndPerturbNoise,
    "input_additive_gaussian": InputAdditiveGaussianNoise,
    "input_multiplicative_gaussian": InputMultiplicativeGaussianNoise,
    "input_additive_uniform": InputAdditiveUniformNoise,
    "input_multiplicative_uniform": InputMultiplicativeUniformNoise,
    "input_ods": InputODSNoise,
    "input_augmix": InputAugMixNoise,
    "input_random_crop_horizontal_flip": InputRandomCropHorizontalFlipNoise,
    "input_target_mixup": InputTargetMixUpNoise,
    "input_target_cmixup": InputTargetCMixUpNoise,
    "activation_additive_gaussian": ActivationAdditiveGaussianNoiseLayer,
    "activation_multiplicative_gaussian": ActivationMultiplicativeGaussianNoiseLayer,
    "activation_additive_uniform": ActivationAdditiveUniformNoiseLayer,
    "activation_multiplicative_uniform": ActivationMultiplicativeUniformNoiseLayer,
    "activation_dropout": ActivationDropoutNoiseLayer,
    "weight_additive_gaussian": WeightAdditiveGaussianNoiseLayer,
    "weight_multiplicative_gaussian": WeightMultiplicativeGaussianNoiseLayer,
    "weight_additive_uniform": WeightAdditiveUniformNoiseLayer,
    "weight_multiplicative_uniform": WeightMultiplicativeUniformNoiseLayer,
    "weight_dropconnect": WeightDropconnectNoiseLayer,
}
