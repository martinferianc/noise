
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
from matplotlib.colors import LinearSegmentedColormap

from sklearn.decomposition import PCA
from natsort import natsorted


plt.style.use(["science", "ieee"])


def vectorize_weights_(weights: List[torch.Tensor]) -> np.ndarray:
    """Vectorize the weights of a single model."""
    vec = [w.flatten().cpu().detach().numpy() for w in weights]
    vec = np.hstack(vec)
    return vec


def vectorize_weight_list_(weight_list: List[List[torch.Tensor]]) -> np.ndarray:
    """Vectorize the weights of a list of models."""
    vec_list = []
    for weights in weight_list:
        vec_list.append(vectorize_weights_(weights))
    weight_matrix = np.column_stack(vec_list)
    return weight_matrix


def shape_weight_matrix_like_(
    weight_matrix: np.ndarray, example: List[torch.Tensor]
) -> List[List[torch.Tensor]]:
    """Reshape the weight matrix to fit the network architecture."""
    weight_vecs = np.hsplit(weight_matrix, weight_matrix.shape[1])
    sizes = [v.numel() for v in example]
    shapes = [v.shape for v in example]
    weight_list = []
    for net_weights in weight_vecs:
        vs = np.split(net_weights, np.cumsum(sizes))[:-1]
        vs = [v.reshape(s) for v, s in zip(vs, shapes)]
        weight_list.append(vs)
    return weight_list


def get_path_components_(
    weight_list: List[List[torch.Tensor]], n_components: int = 2
) -> List[List[torch.Tensor]]:
    # Vectorize network weights
    weight_matrix = vectorize_weight_list_(weight_list)
    # Create components
    pca = PCA(n_components=n_components, whiten=True)
    components = pca.fit_transform(weight_matrix)
    # Reshape to fit network
    example = weight_list[0]
    weight_list = shape_weight_matrix_like_(components, example)
    return pca, weight_list


class PCACoordiantes:
    """This class is used to store the coordinates based on a PCA."""

    def __init__(
        self, weight_list: List[List[torch.Tensor]], n_components: int = 2
    ) -> None:
        self.pca, self.components = get_path_components_(
            weight_list, n_components=n_components
        )
        self.set_origin(weight_list[-1])

    def __call__(self, a: float, b: float) -> List[torch.Tensor]:
        """Return the interpolated weights."""
        return [
            a * v0 + b * v1 + w for v0, v1, w in zip(self._v0, self._v1, self._weights)
        ]

    def normalize_weights(
        self, weights: List[nn.Parameter], weights_ref: List[nn.Parameter]
    ) -> List[torch.Tensor]:
        """Normalize the weights to have the same norm as the weights_ref."""
        return [
            torch.tensor(w) * torch.norm(w_ref) / torch.norm(torch.tensor(w))
            for w, w_ref in zip(weights, weights_ref)
        ]

    def set_origin(self, weights: List[nn.Parameter]) -> None:
        """Set the origin of the coordinate system."""
        self._weights = weights
        self._v0 = self.normalize_weights(self.components[0], weights)
        self._v1 = self.normalize_weights(self.components[1], weights)


def weights_to_coordinates(coords: PCACoordiantes, weights: List[nn.Parameter]):
    """Project the training path onto the first two principal components
    using the pseudoinverse."""
    components = [coords._v0, coords._v1]
    comp_matrix = vectorize_weight_list_(components)
    # the pseudoinverse
    comp_matrix_i = np.linalg.pinv(comp_matrix)
    # the origin vector
    w_c = vectorize_weights_(weights[-1])
    # center the weights on the training path and project onto components
    coord_path = np.array(
        [comp_matrix_i @ (vectorize_weights_(w) - w_c) for w in weights]
    )
    return coord_path


def landscape_2d_path(
    model_dir: str,
    save_dir: str,
    dataset: str,
    architecture: str,
    iterations: int,
    augmentations: Optional[List[str]] = None,
    num_points: int = 10,
    gpu: int = 0,
    hyperparameters: Dict = {},
    data_root_dir: str = "~/.torch",
    seed: int = 0,
    debug: bool = False,
    detailed: bool = False,
) -> None:
    """This script is going to visualise the 2D metric landscape of the training trajectory.

    It saves all the interpolated models, metrics and predictions and plots.
    """
    # Create experiment director
    os.makedirs(save_dir, exist_ok=True)

    results = {}  # Here we store the metrics for all the models per alpha
    coordinates = None
    path = None

    if not os.path.exists(os.path.join(save_dir, "results.pt")):
        hyperparameters = get_hyperparameters(
            dataset, architecture, hyperparameters)

        # We are going to load in all the weights of the training trajectory from the args.model_dir
        # and store them in a list, the files begin with "weights_"
        model_files = [
            f for f in os.listdir(model_dir) if f.startswith("weights_epoch_")
        ]
        model_files = natsorted(model_files, key=lambda y: y.lower())
        # Append `initial` and `final` to the list
        model_files = ["weights_initial.pt"] + \
            model_files + ["weights_final.pt"]
        weights_list = []
        for model_file in model_files:
            model, _ = utils.load_complete_model(
                dataset,
                architecture,
                hyperparameters,
                os.path.join(model_dir, model_file),
            )
            weights_list.append(list(model.parameters()))

        coordinates = PCACoordiantes(weights_list, n_components=2)
        path = weights_to_coordinates(coordinates, weights_list)

        # Adjust the start and end values of alpha and beta according to the path
        # +- 30% of the range
        ALPHA = [min(path[:, 0]), max(path[:, 0])]
        ALPHA = [
            ALPHA[0] - 0.3 * (ALPHA[1] - ALPHA[0]),
            ALPHA[1] + 0.3 * (ALPHA[1] - ALPHA[0]),
        ]
        BETA = [min(path[:, 1]), max(path[:, 1])]
        BETA = [
            BETA[0] - 0.3 * (BETA[1] - BETA[0]),
            BETA[1] + 0.3 * (BETA[1] - BETA[0]),
        ]

        alphas = np.linspace(ALPHA[0], ALPHA[1], num_points)
        betas = np.linspace(BETA[0], BETA[1], num_points)

        for alpha in alphas:
            for beta in betas:
                model_copy, _ = utils.load_complete_model(
                    dataset,
                    architecture,
                    hyperparameters,
                    os.path.join(model_dir, "weights_final.pt"),
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
        utils.save_pickle(path, os.path.join(save_dir, "path.pt"))
    else:
        results = load_pickle(os.path.join(save_dir, "results.pt"))
        path = load_pickle(os.path.join(save_dir, "path.pt"))

    # Plot the metrics
    task, _, _, levels, augmentation_types = get_dataset_options(dataset)
    metric_labels = (
        ClassificationMetric.metric_labels
        if task == "classification"
        else RegressionMetric.metric_labels
    )
    datasets = ["train", "valid", "test"]

    metric_combinations = []
    if task == "classification":
        metric_combinations = [
            ("error", "nll"), ("error", "ece"), ("nll", "ece")]
    elif task == "regression":
        metric_combinations = [("mse", "nll")]

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
                fig, ax = plt.subplots(figsize=(7, 5))
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
                ax.clabel(contour_lines, inline=True, fontsize=12, colors="w")
                cbar = fig.colorbar(c, ax=ax)
                cbar.ax.set_ylabel(METRICS_MAPPING[metric_label], rotation=90)

                # Plot the path
                ax.scatter(path[:, 0], path[:, 1], c="k", s=10, zorder=10)
                # Plot the start and end points
                ax.scatter(path[0, 0], path[0, 1], c="#0000ff",
                           marker="*", s=100, zorder=10, label="Start weights")
                ax.scatter(path[-1, 0], path[-1, 1], c="k",
                           marker="*", s=100, zorder=10, label="End weights")
                ax.set_xlabel("$\\alpha$")
                ax.set_ylabel("$\\beta$")
                # Put the legend above the plot
                ax.legend(loc="upper center",
                          bbox_to_anchor=(0.5, 1.1), ncol=2)

                fig.savefig(
                    os.path.join(save_dir, f"{dataset}_{metric_label}.pdf"),
                    bbox_inches="tight",
                )
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
                      fontsize=12, colors="#00B20F")
            cbar = fig.colorbar(c, ax=ax)
            cbar.ax.set_ylabel(METRICS_MAPPING[metric_label_b], rotation=90)
            # Plot the path
            # Set the colors as a gradient from black to blue
            colors = np.linspace(0, 1, len(path))
            # Define custom colormap
            cmap = LinearSegmentedColormap.from_list(
                "custom blue", ["#0000ff", "#000000"], N=len(path))
            ax.scatter(path[:, 0], path[:, 1], c=colors,
                       cmap=cmap, s=10, zorder=10)
            # Plot the start and end points
            ax.scatter(path[0, 0], path[0, 1], c="#0000ff",
                       marker="*", s=100, zorder=10, label="Start weights")
            ax.scatter(path[-1, 0], path[-1, 1], c="k",
                       marker="*", s=100, zorder=10, label="End weights")
            ax.set_xlabel("$\\alpha$")
            ax.set_ylabel("$\\beta$")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2)
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
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to the model for which to wisualise the 2D loss landscape",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory where to save the results",
    )
    parser.add_argument("--dataset", type=str, help="Dataset to use")
    parser.add_argument("--architecture", type=str, help="Architecture to use")
    parser.add_argument(
        "--iterations", type=int, default=None, help="Number of batches to evaluate on"
    )
    parser.add_argument(
        "--augmentations",
        type=str,
        nargs="+",
        default=None,
        help="Augmentations to evaluate on",
    )
    parser.add_argument(
        "--num_points", type=int, default=10, help="Number of points to interpolate"
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
    landscape_2d_path(**vars(args))
