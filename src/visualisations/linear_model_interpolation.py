# Adapted from https://mathformachines.com/posts/visualizing-the-loss-landscape/

from typing import Dict, List, Optional
import argparse
import json
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

import src.utils as utils
from src.data import get_dataset_options
from src.evaluation import evaluate
from src.evaluation.metrics import (
    METRICS_MAPPING,
    DATASET_MAPPING,
    ClassificationMetric,
    RegressionMetric,
)
from src.training.hyperparameters import get_hyperparameters
from src.utils import load_pickle

plt.style.use(["science", "ieee"])

ALPHAS = np.linspace(0, 1, 20)
tab20 = plt.get_cmap("tab20")
colors = tab20(np.linspace(0, 1, len(ALPHAS)))


def _interpolate_weights(
    model_a: nn.Module, model_b: nn.Module, alpha: float
) -> nn.Module:
    """Interpolate between the weights of two models.

    The interpolation is done in-place with respect to model_a.
    The batch normalisation parameters including the running mean and variance are interpolated as well.

    Proof: "In the 1D linear interpolation methods, the Batch Normalization (BN)
    parameters including the “running mean” and “running variance” need to be considered as part of θ. If these
    parameters are not considered, then it is not possible to reproduce the exact loss values for both minimizers."
    """
    # First interpolate all the parameters
    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert (
            name_a == name_b
        ), f"The two models have different parameters {name_a} != {name_b}"
        param_a.data = alpha * param_a.data + (1 - alpha) * param_b.data

    # Then interpolate the batch normalisation parameters
    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_buffers(), model_b.named_buffers()
    ):
        assert (
            name_a == name_b
        ), f"The two models have different buffers {name_a} != {name_b}"
        param_a.data = alpha * param_a.data + (1 - alpha) * param_b.data

    return model_a


def linear_model_interpolation(
    model_a_path: str,
    model_b_path: str,
    save_dir: str,
    dataset: str,
    architecture: str,
    iterations: int,
    augmentations: Optional[List[str]] = None,
    gpu: int = 0,
    hyperparameters: Dict = {},
    data_root_dir: str = "~/.torch",
    seed: int = 0,
    debug: bool = False,
    detailed: bool = False,
) -> None:
    """This script is going to visualise the linear interpolation between two models.

    It first loads the initial and final model and then interpolates between them
    given the alpha values.

    It saves all the interpolated models, metrics and predictions and plots:
        - The metrics for all the models
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    results = {}  # Here we store the metrics for all the models per alpha

    # Check if load is None and the results are not already computed
    if not os.path.exists(os.path.join(save_dir, "results.pt")):
        hyperparameters = get_hyperparameters(
            dataset, architecture, hyperparameters)

        model_b, _ = utils.load_complete_model(
            dataset, architecture, hyperparameters, model_b_path
        )

        for alpha in ALPHAS:
            # Interpolate between the two models
            # Load a new model
            model, _ = utils.load_complete_model(
                dataset, architecture, hyperparameters, model_a_path
            )
            # Linearly interpolate between the two models and put the result in the new model
            _interpolate_weights(model, model_b, alpha)

            model = utils.model_to_gpu(model, gpu)

            # Evaluate the model
            results[alpha], _ = evaluate(
                model=model,
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
    datasets = ["train", "valid", "test"]

    metric_combinations = []
    if task == "classification":
        metric_combinations = [
            ("error", "nll"), ("error", "ece"), ("nll", "ece")]
    elif task == "regression":
        metric_combinations = [("mse", "nll")]

    dataset_combinations = [("train", "test"), ("test", "augmentations")]

    augmentation_datasets = []
    for i, augmentation in enumerate(augmentation_types):
        if augmentations is not None and augmentation not in augmentations:
            continue
        for level in levels[i]:
            datasets.append("{}_{}".format(augmentation, level))
            augmentation_datasets.append("{}_{}".format(augmentation, level))

    datasets.append("augmentations")

    # Plot each metric and dataset individually
    if detailed:
        for dataset in datasets:
            for metric_label in metric_labels:
                fig, ax = plt.subplots(figsize=(2, 2))
                values = get_values(
                    results, dataset, metric_label, augmentation_datasets, ALPHAS
                )
                ax.plot(ALPHAS, values, marker="o", color="black")
                ax.set_xlabel("$\\alpha$")
                ax.set_ylabel(METRICS_MAPPING[metric_label])
                ax.grid()
                fig.savefig(
                    os.path.join(save_dir, f"{dataset}_{metric_label}.pdf"),
                    bbox_inches="tight",
                )
                plt.close(fig)

    # Plot combinations of metrics and datasets
    # The y-axis on the left side is for the first metric and the y-axis on the right side is for the second metric
    for dataset_combination in dataset_combinations:
        for metric_combination in metric_combinations:
            fig, ax1 = plt.subplots(figsize=(3, 3))
            ax2 = ax1.twinx()
            values1 = get_values(
                results,
                dataset_combination[0],
                metric_combination[0],
                augmentation_datasets,
                ALPHAS,
            )
            values2 = get_values(
                results,
                dataset_combination[1],
                metric_combination[0],
                augmentation_datasets,
                ALPHAS,
            )
            values3 = get_values(
                results,
                dataset_combination[0],
                metric_combination[1],
                augmentation_datasets,
                ALPHAS,
            )
            values4 = get_values(
                results,
                dataset_combination[1],
                metric_combination[1],
                augmentation_datasets,
                ALPHAS,
            )
            ax1.plot(
                ALPHAS,
                values1,
                marker="o",
                color="#AC1E25",
                label=DATASET_MAPPING[dataset_combination[0]],
            )
            ax1.plot(
                ALPHAS,
                values2,
                marker="^",
                color="#AC1E25",
                label=DATASET_MAPPING[dataset_combination[1]],
                linestyle="--",
            )
            ax2.plot(
                ALPHAS,
                values3,
                marker="x",
                color="#00B20F",
                label=DATASET_MAPPING[dataset_combination[0]],
                linestyle=":",
            )
            ax2.plot(
                ALPHAS,
                values4,
                marker="s",
                color="#00B20F",
                label=DATASET_MAPPING[dataset_combination[1]],
                linestyle="-.",
            )
            ax1.set_xlabel("$\\alpha$")
            ax1.set_ylabel(METRICS_MAPPING[metric_combination[0]])
            ax2.set_ylabel(METRICS_MAPPING[metric_combination[1]])
            ax1.grid()
            ax2.grid(linestyle="dotted")
            # Move the legend above the plot and put them horizontally next to each other
            ax1.legend(loc="upper center", bbox_to_anchor=(0.175, 1.1), ncol=2)
            ax2.legend(loc="upper center", bbox_to_anchor=(0.875, 1.1), ncol=2)
            fig.savefig(
                os.path.join(
                    save_dir,
                    f"{dataset_combination[0]}_{dataset_combination[1]}_{metric_combination[0]}_{metric_combination[1]}.pdf",
                ),
                bbox_inches="tight",
            )


def get_values(
    results: Dict,
    dataset: str,
    metric_label: str,
    augmentation_datasets: List[str],
    alphas: List[float],
) -> List[float]:
    """This is a helpepr function to get the values for a specific metric and dataset.

    If the dataset is `augmentations` take the average across all the augmentations.
    """
    if dataset == "augmentations":
        values = []
        for alpha in alphas:
            alpha_values = []
            for augmentation_dataset in augmentation_datasets:
                alpha_values.append(
                    results[alpha][metric_label][augmentation_dataset])
            values.append(np.mean(alpha_values))
    else:
        values = [results[alpha][metric_label][dataset] for alpha in alphas]
    if metric_label in ["ece", "error"]:
        values = [100 * value for value in values]

    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise metrics from a log file")
    parser.add_argument("--model_a_path", type=str,
                        help="Path to the initial model")
    parser.add_argument("--model_b_path", type=str,
                        help="Path to the final model")
    parser.add_argument(
        "--experiment_dir", type=str, help="Path to the experiment directory"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="In case predictions and results are already computed, load them from this directory",
    )
    parser.add_argument("--dataset", type=str, help="Dataset to use")
    parser.add_argument("--architecture", type=str, help="Architecture to use")
    parser.add_argument(
        "--iterations", type=int, default=None, help="Number of batches to evaluate on"
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ids")
    parser.add_argument("--hyperparameters", type=json.loads, default={})
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="~/.torch",
        help="Root directory where the data is stored",
    )
    parser.add_argument(
        "--augmentations",
        type=str,
        nargs="+",
        default=None,
        help="Augmentations to evaluate on",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Whether to plot the metrics for each dataset and metric individually",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument(
        "--debug", action="store_true", help="whether we are currently debugging"
    )

    args = parser.parse_args()
    linear_model_interpolation(**vars(args))
