from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import logging

from src.evaluation.metrics import metric_factory
from src.data import get_dataloaders, get_dataset_options


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: str,
    batch_size: int,
    valid_portion: float,
    test_portion: float,
    seed: int,
    data_root_dir: str,
    validation_only: bool = False,
    cache_predictions: bool = False,
    debug: bool = False,
    iterations: Optional[int] = None,
    augmentations: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], Optional[Dict[str, torch.Tensor]]]:
    """Evaluate a model on a dataset and all splits.

    Performs evaluation on a dataset and records the results in a dictionary.
    Also enables the caching of the predictions.

    Args:
        model (nn.Module): Model to evaluate.
        dataset (str): Dataset to evaluate on.
        batch_size (int): Batch size.
        valid_portion (float): Portion of the training set to use for validation.
        test_portion (float): Portion of the training set to use for testing.
        seed (int): Random seed.
        data_root_dir (str): Root directory for the dataset.
        validation_only (bool): If True, only evaluate on the validation set.
        cache_predictions (bool): If True, cache the predictions.
        debug (bool): If True, only evaluate on a single batch.
        iterations (int): Number of batches to evaluate on.
        augmentations (Optional[List[str]]): List of augmentation names to evaluate on.
    """
    results_dict = {}
    predictions_dict = None if not cache_predictions else {}

    task, _, _, levels, augmentation_types = get_dataset_options(dataset)
    train_loader, valid_loader, test_loader = get_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        valid_portion=valid_portion,
        test_portion=test_portion,
        seed=seed,
        data_root_dir=data_root_dir,
    )

    if augmentations is not None:
        new_levels = []
        new_augmentation_types = []
        for i in range(len(augmentation_types)):
            if augmentation_types[i] in augmentations:
                new_levels.append(levels[i])
                new_augmentation_types.append(augmentation_types[i])
        assert len(augmentations) == len(
            new_augmentation_types
        ), f"The augmentations names {augmentations} are misspecified. The available augmentations are {augmentation_types}."

        levels = new_levels
        augmentation_types = new_augmentation_types

    if validation_only:
        logging.info("### Evaluating model on valid ###")
        evaluate_loader(
            model=model,
            results_dict=results_dict,
            predictions_dict=predictions_dict,
            loader=valid_loader,
            label="valid",
            task=task,
            debug=debug,
            iterations=iterations,
        )
        return results_dict, predictions_dict

    logging.info("### Evaluating model on train ###")
    evaluate_loader(
        model=model,
        results_dict=results_dict,
        predictions_dict=predictions_dict,
        loader=train_loader,
        label="train",
        task=task,
        debug=debug,
        iterations=iterations,
    )

    logging.info("### Evaluating model on valid ###")
    evaluate_loader(
        model=model,
        results_dict=results_dict,
        predictions_dict=predictions_dict,
        loader=valid_loader,
        label="valid",
        task=task,
        debug=debug,
        iterations=iterations,
    )

    logging.info("### Evaluating model on test ###")
    evaluate_loader(
        model=model,
        results_dict=results_dict,
        predictions_dict=predictions_dict,
        loader=test_loader,
        label="test",
        task=task,
        debug=debug,
        iterations=iterations,
    )
    for i, augmentation in enumerate(augmentation_types):
        for level in levels[i]:
            loaders = get_dataloaders(
                dataset=dataset,
                batch_size=batch_size,
                valid_portion=valid_portion,
                test_portion=test_portion,
                seed=seed,
                augmentation=augmentation,
                level=level,
                data_root_dir=data_root_dir,
            )
            _, _, test_loader = loaders
            logging.info(
                "### Evaluating model on test augmentation: {} level: {} ###".format(
                    augmentation, level
                )
            )
            evaluate_loader(
                model=model,
                results_dict=results_dict,
                predictions_dict=predictions_dict,
                loader=test_loader,
                label="{}_{}".format(augmentation, level),
                task=task,
                debug=debug,
                iterations=iterations,
            )
    return results_dict, predictions_dict


def evaluate_loader(
    model: nn.Module,
    results_dict: Dict[str, float],
    predictions_dict: Optional[Dict[str, torch.Tensor]],
    loader: torch.utils.data.DataLoader,
    label: str,
    task: str,
    debug: bool = False,
    iterations: Optional[int] = None,
) -> None:
    """Step through a loader and compute the metrics and cache the outputs."""
    metric = metric_factory(
        task=task, output_size=model.output_size, writer=None, log=None
    )
    outputs = torch.empty((0, model.output_size),
                          dtype=torch.float32, device="cpu")
    for i, (input, target) in enumerate(loader):
        input = input.to(next(model.parameters()).device)
        target = target.to(next(model.parameters()).device)
        output = model(input)
        metric.update(output=output, target=target)
        outputs = torch.cat((outputs, output.detach().cpu()), dim=0)
        if (i >= 2 and debug) or (iterations is not None and i >= iterations - 1):
            break

    metrics = metric.get_packed()
    for key, value in metrics.items():
        logging.info(
            "### Result metric {}: {}: {} ###".format(label, key, value))
        if results_dict is not None:
            if key not in results_dict:
                results_dict[key] = {}
            results_dict[key][label] = value

    if predictions_dict is not None:
        predictions_dict[label] = outputs
