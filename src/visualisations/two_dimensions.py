# Adapted from https://mathformachines.com/posts/visualizing-the-loss-landscape/

from src.utils import load_pickle
from src.training.hyperparameters import get_hyperparameters
from src.evaluation.metrics import (
    METRICS_MAPPING,
    ClassificationMetric,
    RegressionMetric,
)
from src.evaluation import evaluate
from src.data import get_dataset_options
import src.utils as utils
from typing import Dict, List, Tuple, Optional
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import torch
import torch.nn as nn


plt.style.use(["science", "ieee"])

ALPHA = [-1, 1]
BETA = [-1, 1]


class RandomCoordinates:
    """This class is used to store the random coordinates for the 2D landscape."""

    def __init__(self, weights: List[nn.Parameter], seed: int = 42) -> None:
        self._generator = torch.Generator().manual_seed(seed)
        self._weights = weights
        self._v0 = self.normalize_weights(
            [
                torch.randn(weight.shape, generator=self._generator)
                for weight in weights
            ],
            weights,
        )
        self._v1 = self.normalize_weights(
            [
                torch.randn(weight.shape, generator=self._generator)
                for weight in weights
            ],
            weights,
        )

    def normalize_weights(
        self, weights: List[nn.Parameter], weights_ref: List[nn.Parameter]
    ) -> List[torch.Tensor]:
        """Normalize the weights to have the same norm as the weights_ref."""
        return [
            w * torch.norm(w_ref) / torch.norm(w)
            for w, w_ref in zip(weights, weights_ref)
        ]

    def __call__(self, a: float, b: float) -> List[torch.Tensor]:
        """Return the interpolated weights."""
        return [
            a * v0 + b * v1 + w for v0, v1, w in zip(self._v0, self._v1, self._weights)
        ]


def landscape_2d(
    model_path: str,
    save_dir: str,
    dataset: str,
    architecture: str,
    iterations: int = None,
    augmentations: Optional[List[str]] = None,
    num_points: int = 10,
    limit: float = 1,
    gpu: int = 0,
    hyperparameters: Dict = {},
    data_root_dir: str = "~/.torch",
    seed: int = 42,
    debug: bool = False,
    detailed: bool = False,
) -> None:
    """This script is going to visualise the 2D metric landscape of a model.

    It saves all the interpolated models, metrics and predictions and plots
    """
    # Create experiment directory
    os.makedirs(save_dir, exist_ok=True)

    results = {}  # Here we store the metrics for all the models per alpha

    if not os.path.exists(os.path.join(save_dir, "results.pt")):
        hyperparameters = get_hyperparameters(
            dataset, architecture, hyperparameters)

        model, _ = utils.load_complete_model(
            dataset, architecture, hyperparameters, model_path
        )

        alphas = np.linspace(ALPHA[0], ALPHA[1], num_points) ** 3 * limit
        betas = np.linspace(BETA[0], BETA[1], num_points) ** 3 * limit

        coordinates = RandomCoordinates(list(model.parameters()), seed=seed)

        for alpha in alphas:
            for beta in betas:
                model_copy, _ = utils.load_complete_model(
                    dataset, architecture, hyperparameters, model_path
                )
                # Change the model with respect to the random coordinates
                new_parameters = coordinates(alpha, beta)
                for param, new_param in zip(model_copy.parameters(), new_parameters):
                    param.data = new_param

                model_copy = utils.model_to_gpu(model_copy, gpu)

                # Evaluate the model
                results[(alpha, beta)], _ = evaluate(
                    model=model_copy,
                    dataset=dataset,
                    batch_size=hyperparameters["batch_size"],
                    valid_portion=hyperparameters["valid_portion"],
                    test_portion=hyperparameters["test_portion"],
                    seed=seed,
                    data_root_dir=data_root_dir,
                    validation_only=False,
                    cache_predictions=True,
                    iterations=iterations,
                    augmentations=augmentations,
                    debug=debug,
                )

        # Save the results
        utils.save_pickle(results, os.path.join(save_dir, "results.pt"))
    else:
        results = load_pickle(os.path.join(save_dir, "results.pt"))

    # Plot the metrics
    task, _, _, levels, augmentation_types = get_dataset_options(dataset)
    metric_labels = (
        ClassificationMetric.metric_labels
        if task == "classification"
        else RegressionMetric.metric_labels
    )

    metric_combinations = []
    if task == "classification":
        metric_combinations = [
            ("error", "nll"), ("error", "ece"), ("nll", "ece")]
    elif task == "regression":
        metric_combinations = [("mse", "nll")]

    datasets = ["train", "valid", "test"]

    dataset_combinations = ["train", "valid", "test", "augmentations"]

    augmentation_datasets = []
    for i, augmentation in enumerate(augmentation_types):
        if augmentations is not None and augmentation not in augmentations:
            continue
        for level in levels[i]:
            datasets.append("{}_{}".format(augmentation, level))
            augmentation_datasets.append("{}_{}".format(augmentation, level))

    datasets.append("augmentations")

    # Create a contour plot for each metric
    if detailed:
        for dataset in datasets:
            for metric_label in metric_labels:
                fig, ax = plt.subplots(figsize=(5, 3))
                alphas = [alpha for alpha, _ in results.keys()]
                betas = [beta for _, beta in results.keys()]
                # Make sure to sort the alphas and betas
                alphas = np.sort(np.unique(alphas))
                betas = np.sort(np.unique(betas))
                values = get_values(
                    results,
                    dataset,
                    metric_label,
                    augmentation_datasets,
                    alphas,
                    betas,
                )
                values = values.reshape(len(alphas), len(betas))
                c = ax.contourf(
                    alphas, betas, values, cmap="magma", levels=20, extend="both", alpha=0.7
                )
                ax.grid(True, linestyle="--", linewidth=0.5)
                contour_lines = ax.contour(
                    alphas, betas, values, colors="w", levels=10, linewidths=1
                )
                ax.clabel(contour_lines, inline=True, fontsize=12,  colors="w")
                cbar = fig.colorbar(c, ax=ax, format="%1.2e")
                cbar.ax.set_ylabel(METRICS_MAPPING[metric_label], rotation=90)
                ax.scatter([0], [0], marker="*", color="k",
                           s=100, label="Original weights")
                fig.savefig(
                    os.path.join(save_dir, f"{dataset}_{metric_label}.pdf"),
                    bbox_inches="tight",
                )
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1))
                plt.close(fig)

    # Plot cobinations of metrics with respect to a single dataset
    # Plot countour plots for each metric, metric 1 in red, metric 2 in blue
    # Do not use fill contours
    for dataset in dataset_combinations:
        alphas = [alpha for alpha, _ in results.keys()]
        betas = [beta for _, beta in results.keys()]
        # Make sure to sort the alphas and betas
        alphas = np.sort(np.unique(alphas))
        betas = np.sort(np.unique(betas))
        for metric_label_a, metric_label_b in metric_combinations:
            fig, ax = plt.subplots(figsize=(5, 3))
            values_a = get_values(
                results,
                dataset,
                metric_label_a,
                augmentation_datasets,
                alphas,
                betas,
            )
            values_b = get_values(
                results,
                dataset,
                metric_label_b,
                augmentation_datasets,
                alphas,
                betas,
            )
            values_a = values_a.reshape(len(alphas), len(betas))
            values_b = values_b.reshape(len(alphas), len(betas))
            c = ax.contourf(
                alphas,
                betas,
                values_a,
                cmap="Reds",
                levels=20,
                extend="both",
                alpha=0.7,
            )
            ax.grid(True, linestyle="--", linewidth=0.5)
            contour_lines = ax.contour(
                alphas, betas, values_a, colors="#AC1E25", levels=10, linewidths=1
            )
            ax.clabel(contour_lines, inline=True,
                      fontsize=12, colors="#AC1E25")
            cbar = fig.colorbar(c, ax=ax)
            cbar.ax.set_ylabel(METRICS_MAPPING[metric_label_a], rotation=90)
            c = ax.contourf(
                alphas,
                betas,
                values_b,
                cmap="Greens",
                levels=20,
                extend="both",
                alpha=0.7,
            )
            ax.grid(True, linestyle="--", linewidth=0.5)
            contour_lines = ax.contour(
                alphas, betas, values_b, colors="#00B20F", levels=10, linewidths=1
            )
            ax.clabel(contour_lines, inline=True,
                      fontsize=12,  colors="#00B20F")
            cbar = fig.colorbar(c, ax=ax)
            cbar.ax.set_ylabel(METRICS_MAPPING[metric_label_b], rotation=90)
            ax.scatter([0], [0], marker="*", color="k",
                       s=100, label="Original weights")
            ax.set_xlabel("$\\alpha$")
            ax.set_ylabel("$\\beta$")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1))
            fig.savefig(
                os.path.join(
                    save_dir, f"{dataset}_{metric_label_a}_{metric_label_b}.pdf"
                ),
                bbox_inches="tight",
            )
            plt.close(fig)
            plt.clf()
            plt.cla()


def get_values(
    results: Dict,
    dataset: str,
    metric_label: str,
    augmentation_datasets: List[str],
    alphas: List[float],
    betas: List[float],
) -> List[Tuple[float, float]]:
    """This is a helpepr function to get the values for a specific metric and dataset.

    If the dataset is `augmentations` take the average across all the augmentations.
    """
    values = []
    if dataset == "augmentations":
        values = []
        for alpha in alphas:
            for beta in betas:
                values.append(
                    np.mean(
                        [
                            results[(alpha, beta)
                                    ][metric_label][augmentation_dataset]
                            for augmentation_dataset in augmentation_datasets
                        ]
                    )
                )

        values = np.array(values)
    else:
        values = np.array(
            [
                results[(alpha, beta)][metric_label][dataset]
                for alpha in alphas
                for beta in betas
            ]
        )

    if metric_label in ["ece", "error"]:
        values = 100 * values

    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise metrics from a log file")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory where to save the results",
    )
    parser.add_argument("--dataset", type=str, help="Dataset to use")
    parser.add_argument("--architecture", type=str, help="Architecture to use")
    parser.add_argument(
        "--iterations", type=int, default=None, help="Number of batches to evaluate on"
    )
    parser.add_argument(
        "--num_points", type=int, default=10, help="Number of points to interpolate"
    )
    parser.add_argument(
        "--limit", type=float, default=1, help="Range of the interpolation"
    )
    parser.add_argument(
        "--augmentations",
        type=str,
        nargs="+",
        default=None,
        help="Augmentations to evaluate on",
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ids")
    parser.add_argument("--hyperparameters", type=json.loads, default={})
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="~/.torch",
        help="Root directory where the data is stored",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed")

    parser.add_argument(
        "--debug", action="store_true", help="whether we are currently debugging"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="whether we are currently debugging",
    )
    args = parser.parse_args()
    landscape_2d(**vars(args))
