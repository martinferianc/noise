import sys

sys.path.append("../")

from visualise import visualise_experiment
from src.training.hyperparameters import get_hyperparameters
from src.training import train
import src.utils as utils
import argparse
import json
import logging
import os
import sys
from typing import Optional

from evaluate import evaluate_model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_model(
    args: argparse.Namespace, writer: Optional[utils.SummaryWriter] = None
) -> None:
    logging.info("# Starting training #")
    hyperparameters = get_hyperparameters(
        args.dataset, args.architecture, args.hyperparameters
    )
    logging.info("## Hyperparameters: %s ##" % hyperparameters)

    model, noises = utils.load_complete_model(
        args.dataset, args.architecture, hyperparameters, path=args.load
    )

    # Save the initial model
    if args.save_initial_model:
        utils.save_model(model, args.save, special_info="_initial")

    model = utils.model_to_gpu(model, args.gpu)

    log = None
    if args.save_logs:
        log = {}

    train(args, model, noises, writer=writer,
          log=log, hyperparameters=hyperparameters)

    utils.save_model(model, args.save, special_info="_final")

    if args.save_logs:
        utils.save_pickle(log, os.path.join(args.save, "log.pt"))

    logging.info("# Finished training #")
    logging.info(
        "# Evaluating with respect to default parameters during training #")

    # Save args
    utils.save_pickle(args, os.path.join(args.save, "args.pt"), overwrite=True)

    evaluate_model(args, model, hyperparameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("training")

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="define the dataset and thus task",
    )
    parser.add_argument(
        "--architecture", type=str, default="fc", help="name of the model to use"
    )
    parser.add_argument(
        "--save", type=str, default="./runs", help="experiment directory"
    )
    parser.add_argument("--load", type=str, default=None,
                        help="load model from path")
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="~/.torch",
        help="the default data root directory",
    )
    parser.add_argument("--label", type=str, default="",
                        help="experiment name")

    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--save_initial_model",
        action="store_true",
        help="whether to save the initial model",
    )
    parser.add_argument(
        "--save_every_epoch",
        action="store_true",
        help="whether to save the model every epoch",
    )
    parser.add_argument(
        "--save_logs",
        action="store_true",
        help="whether to save the training logs",
    )
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
        "--debug", action="store_true", help="whether we are currently debugging"
    )
    parser.add_argument(
        "--mute", action="store_true", help="whether to mute the logger"
    )
    parser.add_argument(
        "--visualise",
        action="store_true",
        help="whether to visualise the results",
    )
    parser.add_argument(
        "--evaluation_iterations",
        type=int,
        default=None,
        help="number of batches to evaluate on",
    )
    parser.add_argument(
        "--evaluation_augmentations",
        nargs="+",
        default=None,
        help="augmentations to evaluate on",
    )
    parser.add_argument(
        "--visualisation_iterations",
        type=int,
        default=1,
        help="number of 1000 samples batches to evaluate on for visualisations",
    )
    parser.add_argument(
        "--visualisation_augmentations",
        nargs="+",
        default=None,
        help="augmentations to evaluate on for visualisations",
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ids")
    parser.add_argument("--hyperparameters", type=json.loads, default={})

    args, additional_args = parser.parse_known_args()

    args, writer = utils.parse_args(args)
    train_model(args, writer)

    if args.visualise:
        assert args.save_initial_model and args.save_every_epoch and args.save_logs, f"Visualisation requires all of the following arguments to be set: --save_initial_model --save_every_epoch --save_logs"
        visualise_experiment(
            experiment_folder=args.save,
            iterations=args.visualisation_iterations,
            gpu=args.gpu,
            data_root_dir=args.data_root_dir,
            seed=args.seed,
            augmentations=args.visualisation_augmentations,
            debug=args.debug,
        )
