import argparse
import logging
from typing import Any, Dict, List, Optional

from optuna.trial import Trial
import torch
import torch.nn as nn

from src.data import get_dataloaders, get_dataset_options
from src.training.losses import loss_factory
from src.training.trainer import Trainer
from src.utils import SummaryWriter


def train(
    args: argparse.Namespace,
    model: nn.Module,
    noises: List[nn.Module],
    writer: Optional[SummaryWriter] = None,
    log: Optional[Dict[str, Any]] = None,
    hyperparameters: Optional[Dict[str, float]] = None,
    validate: bool = True,
    trial: Optional[Trial] = None,
) -> None:
    """This function trains a model based on the given arguments.

    Additionally it instantiates the data, loss and regularisers.

    Args:
        args (argparse.Namespace): The arguments to use for training.
        model (nn.Module): The model to train.
        noises (List[nn.Module]): The noises to use for training.
        writer (Optional[SummaryWriter]): The writer to use for logging.
        log (Optional[Dict[str, Any]]): The log to use for logging.
        hyperparameters (Optional[Dict[str, float]]): The hyperparameters to use for training.
        validate (bool): Whether to validate the model during training.
    """
    task, _, _, _, _ = get_dataset_options(args.dataset)
    logging.info(
        "## Start training: Dataset: %s, Architecture: %s, Task: %s ##"
        % (args.dataset, args.architecture, task)
    )

    logging.info("### Preparing criterion ###")
    criterion = loss_factory(task=task)

    logging.info(criterion)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=hyperparameters["lr"], weight_decay=0.0, momentum=0.9
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hyperparameters["epochs"], eta_min=0.0
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        l2_weight=hyperparameters["l2"],
        writer=writer,
        log=log,
        epochs=hyperparameters["epochs"],
        noises=noises,
        gradient_norm_clip=hyperparameters["gradient_norm_clip"],
        task=task,
        args=args,
        verbose=True,
        trial=trial,
    )
    logging.info("## Trained model: %s ##" % model)

    logging.info("### Downloading and preparing data ##")
    train_loader, valid_loader, _ = get_dataloaders(
        dataset=args.dataset,
        batch_size=hyperparameters["batch_size"],
        seed=args.seed,
        valid_portion=hyperparameters["valid_portion"],
        test_portion=hyperparameters["test_portion"],
        data_root_dir=args.data_root_dir,
    )

    if not validate:
        valid_loader = None

    trainer.train_loop(train_loader, valid_loader)
