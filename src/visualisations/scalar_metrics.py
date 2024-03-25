from typing import Union, List, Dict, Optional
import argparse
import os

import matplotlib.pyplot as plt
import scienceplots
import natsort

from src.data import get_dataset_options
from src.evaluation.metrics import (
    METRICS_MAPPING,
    DATASET_MAPPING,
    ClassificationMetric,
    RegressionMetric,
)
from src.utils import load_pickle, save_pickle

plt.style.use(["science", "ieee"])


def visualise_metrics(
    model_dir: Union[str, List[str]],
    save_dir: str,
    labels: Union[str, List[str]],
    dataset: str,
    augmentations: Optional[List[str]] = None,
):
    """This script is going to visualise all the scalar metrics from a log file."""

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    task = get_dataset_options(dataset)[0]
    metric_labels = (
        ClassificationMetric.metric_labels
        if task == "classification"
        else RegressionMetric.metric_labels
    )
    metric_labels = metric_labels + \
        ["l2_weight_complete", "l2_gradient_complete"]
    datasets = ["train", "valid"]

    model_dir = [model_dir] if isinstance(model_dir, str) else model_dir
    labels = [labels] if isinstance(labels, str) else labels

    metric_combinations = []
    if task == "classification":
        metric_combinations = [
            ("error", "nll"), ("error", "ece"), ("nll", "ece")]
    elif task == "regression":
        metric_combinations = [("mse", "nll")]

    dataset_combinations = [("train", "valid")]

    if not os.path.exists(os.path.join(save_dir, "results.pt")):
        experiments = {
            experiment: load_pickle(os.path.join(experiment, "log.pt"))
            for experiment in model_dir
        }
        save_pickle(experiments, os.path.join(save_dir, "results.pt"))
    else:
        experiments = load_pickle(os.path.join(save_dir, "results.pt"))

    # Iterate through all the experiment folders and plot the metrics for all the experiments in one plot
    for dataset in datasets:
        for metric_label in metric_labels:
            fig, ax = plt.subplots(figsize=(5, 5))
            for i, experiment in enumerate(experiments):
                # Get the names of all the iterations
                iterations = experiments[experiment][dataset].keys()
                # Sort the iterations
                iterations = natsort.natsorted(iterations)
                values = get_values(
                    experiments[experiment], dataset, metric_label)
                # For these metrics there is a list of values for each iteration (per batch) we need to flatten the list
                # completely
                if metric_label in ["l2_weight_complete", "l2_gradient_complete"]:
                    values = [item for sublist in values for item in sublist]
                    iterations = [i for i in range(len(values))]
                    xlabel = "Batch"
                else:
                    xlabel = "Epoch"
                ax.plot(iterations, values, label=labels[i])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(METRICS_MAPPING[metric_label])
            ax.legend()
            ax.grid()
            fig.savefig(
                os.path.join(
                    save_dir, "{}_{}_{}.pdf".format(
                        dataset, metric_label, task)
                ),
                bbox_inches="tight",
            )
            plt.close(fig)

    # Plot combinations of metrics and datasets
    # The y-axis on the left side is for the first metric and the y-axis on the right side is for the second metric
    # Do this for each experiment separately, do not put multiple experiments in one plot
    for i, experiment in enumerate(experiments):
        for metric_combination in metric_combinations:
            for dataset_combination in dataset_combinations:
                fig, ax1 = plt.subplots(figsize=(3, 3))
                ax2 = ax1.twinx()
                for k, dataset in enumerate(dataset_combination):
                    for j, metric_label in enumerate(metric_combination):
                        # Get the names of all the iterations
                        iterations = experiments[experiment][dataset].keys()
                        # Sort the iterations
                        iterations = natsort.natsorted(iterations)
                        values = get_values(
                            experiments[experiment], dataset, metric_label)
                        # For these metrics there is a list of values for each iteration (per batch) we need to flatten the list
                        # completely
                        if metric_label in [
                            "l2_weight_complete",
                            "l2_gradient_complete",
                        ]:
                            values = [
                                item for sublist in values for item in sublist]
                            iterations = [i for i in range(len(values))]
                            xlabel = "Batch"
                        else:
                            xlabel = "Epoch"
                        if k == 0 and j == 0:
                            marker = "o"
                            color = "red"
                        if k == 0 and j == 1:
                            marker = "x"
                            color = "blue"
                        if k == 1 and j == 0:
                            marker = "^"
                            color = "red"
                        if k == 1 and j == 1:
                            marker = "s"
                            color = "blue"
                        if metric_label == metric_combination[0]:
                            ax1.plot(
                                iterations, values, label=DATASET_MAPPING[dataset], marker=marker, color=color)
                        else:
                            ax2.plot(
                                iterations, values, label=DATASET_MAPPING[dataset], marker=marker, color=color)
                ax1.set_xlabel(xlabel)
                ax1.set_ylabel(
                    METRICS_MAPPING[metric_combination[0]], color="red")
                ax2.set_ylabel(
                    METRICS_MAPPING[metric_combination[1]], color="blue")
                ax1.grid()
                ax2.grid(linestyle="dotted")
                # Move the legend above the plot
                ax1.legend(loc="upper center",
                           bbox_to_anchor=(0.175, 1.1), ncol=2)
                ax2.legend(loc="upper center",
                           bbox_to_anchor=(0.875, 1.1), ncol=2)
                fig.savefig(
                    os.path.join(
                        save_dir,
                        "{}_{}_{}_{}.pdf".format(
                            i,
                            metric_combination[0],
                            metric_combination[1],
                            task,
                        ),
                    ),
                    bbox_inches="tight",
                )


def get_values(results: Dict, dataset: str, metric_label: str) -> List[float]:
    """This is a helpepr function to get the values for a specific metric and dataset."""

    values = []
    for iteration in results[dataset]:
        values.append(results[dataset][iteration][metric_label])

    if metric_label in ["ece", "error"]:
        values = [100 * value for value in values]

    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise metrics from a log file")
    parser.add_argument(
        "--model_dir", type=str, nargs="+", help="Path to the experiment folders"
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", help="Labels for the experiments"
    )
    parser.add_argument(
        "--save_dir", type=str, help="Path to the save directory", default="./"
    )
    parser.add_argument(
        "--augmentations",
        type=str,
        nargs="+",
        default=None,
        help="Augmentations to evaluate on",
    )

    parser.add_argument("--dataset", type=str, help="Dataset to use")
    args = parser.parse_args()
    visualise_metrics(
        model_dir=args.model_dir,
        save_dir=args.save_dir,
        labels=args.labels,
        dataset=args.dataset,
    )

# python3 scalar_metrics.py --model_dirs ./../experiments/runs/regression_energy-fc-20230920-191653/ --labels vanilla --dataset regression_energy --save_dir ./runs/
