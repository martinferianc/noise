from typing import Dict
import torch.nn as nn

from src.data import get_dataset_options
from src.models.fc import FC
from src.models.resnet import ResNet
from src.models.global_pooling_cnn import GlobalPoolingCNN
from src.models.transformer import Transformer


def model_factory(
    dataset: str, architecture: str, hyperparameters: Dict[str, float]
) -> nn.Module:
    """This function creates a model based on the given dataset, architecture and hyperparameters."""
    _, input_size, output_size, _, _ = get_dataset_options(dataset)
    model = None

    if architecture == "fc":
        model = FC(
            input_size=input_size,
            output_size=output_size,
            planes=hyperparameters["planes"],
        )
    elif architecture == "resnet":
        model = ResNet(
            output_size=output_size,
            layers=hyperparameters["layers"],
            planes=hyperparameters["planes"],
            strides=hyperparameters["strides"],
        )
    elif architecture == "global_pooling_cnn":
        model = GlobalPoolingCNN(
            output_size=output_size,
            planes=hyperparameters["planes"],
            input_size=input_size,
        )
    elif architecture == "transformer":
        model = Transformer(
            input_size=input_size,
            output_size=output_size,
            dim=hyperparameters["dim"],
            depth=hyperparameters["depth"],
            heads=hyperparameters["heads"],
            mlp_dim=hyperparameters["mlp_dim"],
        )
    else:
        raise ValueError("Unknown architecture: {}".format(architecture))

    return model
