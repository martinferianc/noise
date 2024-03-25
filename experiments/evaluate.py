from typing import Dict, Optional
import sys
import argparse
import logging
import json
import torch.nn as nn
import os

sys.path.append("../")

from src.evaluation import evaluate
import src.utils as utils
from src.training.hyperparameters import get_hyperparameters


def evaluate_model(
    args: argparse.Namespace,
    model: Optional[nn.Module] = None,
    hyperparameters: Optional[Dict[str, float]] = None,
) -> None:
    """Evaluates a model on a given dataset."""

    logging.info("# Starting Evaluation #")
    if hyperparameters is None:
        hyperparameters = get_hyperparameters(
            args.dataset,
            args.architecture,
            overwrite_hyperparameters=args.hyperparameters,
        )

    if model is None:
        assert args.load is not None, f"Please specify a model to load, got {args.load}"
        model, _ = utils.load_complete_model(args.dataset, args.architecture, hyperparameters, path=args.load)

    model = utils.model_to_gpu(model, args.gpu)

    logging.info("### Setting the model to evaluation mode ###")
    model.eval()

    logging.info("## Model evaluation ##")
    results, predictions = evaluate(
        model,
        args.dataset,
        batch_size=hyperparameters["batch_size"],
        valid_portion=hyperparameters["valid_portion"],
        test_portion=hyperparameters["test_portion"],
        seed=args.seed,
        data_root_dir=args.data_root_dir,
        validation_only=args.validation_only,
        augmentations=args.evaluation_augmentations,
        iterations=args.evaluation_iterations,
        debug=args.debug,
    )
    logging.info("## Evaluation Done ##")

    utils.save_pickle(results, os.path.join(args.save, "results.pt"), overwrite=True)
    # save it also to a separate directory on the distributed disk for easy access
    result_summaries_directory = os.path.join("result_summaries", args.label)
    if not os.path.exists(result_summaries_directory):
        os.makedirs(result_summaries_directory)
    utils.save_pickle(results, os.path.join(result_summaries_directory, "results.pt"), overwrite=True)

    if args.save_predictions:
        utils.save_pickle(
            predictions, os.path.join(args.save, "predictions.pt"), overwrite=True
        )
        
    # Save args
    utils.save_pickle(args, os.path.join(args.save, "args.pt"), overwrite=False)
        
    return results, predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluation")

    parser.add_argument(
        "--dataset", type=str, default="cifar", help="define the dataset and thus task"
    )
    parser.add_argument(
        "--architecture", type=str, default="fc", help="name of the architecture"
    )


    parser.add_argument("--save", type=str, default="./runs", help="experiment directory")
    parser.add_argument("--data_root_dir", type=str, default="~/.torch", help="the default data root directory")
    parser.add_argument("--label", type=str, default="", help="experiment name")
    parser.add_argument("--load", type=str, default="./runs", help="experiment directory")
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="whether to save the testing predictions",
    )
    parser.add_argument(
        "--validation_only",
        action="store_true",
        help="whether to only evaluate on the validation set",
    )
    parser.add_argument(
        "--evaluation_augmentations",
        nargs="+",
        default=None,
        help="augmentations to evaluate on",
    )
    parser.add_argument(
        "--evaluation_iterations",
        type=int,
        default=None,
        help="number of batches to evaluate on",
    )
    
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--debug", action="store_true", help="whether we are currently debugging"
    )
    parser.add_argument(
        "--mute", action="store_true", help="whether we are currently debugging"
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ids")
    parser.add_argument("--hyperparameters", type=json.loads, default={})

    args = parser.parse_args()
    args, _ = utils.parse_args(args)
    evaluate_model(args)
