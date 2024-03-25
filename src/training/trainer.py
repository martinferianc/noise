import argparse
import logging
from typing import Dict, List, Optional, Tuple

import optuna
from optuna.trial import Trial
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils as utils
from src.evaluation.metrics import metric_factory
from src.noises.base import Noise


class Trainer:
    """This class is responsible for training a model.

    It combines together the model, the criterion, the optimizer and the data loader.

    Args:
        model (torch.nn.Module): The model to train.
        criterion (torch.nn.Module): The criterion to use for training.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to use for training.
        epochs (int): The number of epochs to train for.
        l2_weight (float): The weight of the L2 regulariser.
        noises (List[Noise], optional): The noises to use for training. Defaults to None.
        gradient_norm_clip (float, optional): The gradient norm to clip to. Defaults to 5.0.

        writer (torch.utils.tensorboard.SummaryWriter, optional): The tensorboard writer to use for training. Defaults to None.
        args (argparse.Namespace, optional): The arguments to use for training. Defaults to None.
        verbose (bool, optional): Whether to print training information. Defaults to True.
        task (str, optional): The task to train for. Defaults to "classification".
        printing_frequency (int, optional): The frequency of logging. Defaults to 50.
        log (Dict[str, List[float]], optional): The log to use for saving the training information. Defaults to None.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        epochs: int,
        l2_weight: float,
        noises: Optional[List[Noise]] = None,
        gradient_norm_clip: float = 5.0,
        writer: Optional[torch.utils.tensorboard.SummaryWriter] = None,
        args: Optional[argparse.Namespace] = None,
        verbose: bool = True,
        task: str = "classification",
        printing_frequency: int = 50,
        log: Optional[Dict[str, List[float]]] = None,
        trial: Optional[Trial] = None,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.l2_weight = l2_weight
        self.noises = noises
        self.gradient_norm_clip = gradient_norm_clip

        self.writer = writer
        self.args = args
        self.verbose = verbose
        self.printing_frequency = printing_frequency
        self.save_every_epoch = args.save_every_epoch
        self.task = task
        self.trial = trial

        self.train_metrics = metric_factory(
            task=self.task,
            output_size=self.model.output_size,
            writer=self.writer,
            log=log,
        )
        self.valid_metrics = metric_factory(
            task=self.task,
            output_size=self.model.output_size,
            writer=self.writer,
            log=log,
        )

        self.writer = writer
        self.epoch = -1
        self.iteration = 0

    def train_loop(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Tuple[float, float]:
        """This function is responsible for the training loop.

        Args:
            train_loader (torch.utils.data.DataLoader): The training data loader.
            valid_loader (torch.utils.data.DataLoader): The validation data loader.
        """
        for epoch in range(self.epochs):
            self.epoch = epoch

            if self.noises is not None and epoch < int(self.epochs * 0.75):
                for noise in self.noises:
                    noise.pre_epoch(epoch)

            if epoch >= 1 and self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.lr_scheduler is None:
                self.lr_coeff = self.optimizer.param_groups[0]["lr"]
            else:
                self.lr_coeff = self.lr_scheduler.get_lr()[0]

            if self.writer is not None:
                self.writer.add_scalar("train/lr", self.lr_coeff, epoch)

            if self.verbose:
                logging.info(
                    "### Epoch: [%d/%d], Learning rate: %e",
                    epoch + 1,
                    self.epochs,
                    self.lr_coeff,
                )

            self.train(train_loader)

            if self.verbose:
                logging.info("#### Train | %s ####",
                             self.train_metrics.get_str())

            self.train_metrics.scalar_logging("train", epoch)

            if valid_loader is not None:
                self.infer(valid_loader)
                if self.verbose:
                    logging.info("#### Valid | %s ####",
                                 self.valid_metrics.get_str())
                self.valid_metrics.scalar_logging("valid", epoch)

            if self.noises is not None and epoch < int(self.epochs * 0.75):
                for noise in self.noises:
                    noise.post_epoch(epoch)

            if self.save_every_epoch:
                utils.save_model(
                    self.model,
                    self.args.save,
                    special_info="_epoch_%d" % (epoch + 1),
                )

            if valid_loader is not None:
                if self.trial is not None:
                    self.trial.report(
                        self.valid_metrics.nll.compute(), epoch + 1)
                    if self.trial.should_prune():
                        raise optuna.TrialPruned()

    def _step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], train: bool = True
    ) -> None:
        """This function is responsible for a single step of the training loop."""
        loss = 0.0
        l2_loss = 0.0
        l2_weight = 0.0
        l2_gradient = 0.0

        (input, target) = batch

        input = input.to(next(self.model.parameters()).device)
        target = target.to(next(self.model.parameters()).device)
        if self.task == "classification":
            target_one_hot = F.one_hot(target, self.model.output_size).float()
        else:
            target_one_hot = target

        if train and self.noises is not None:
            for noise in self.noises:
                input, target_one_hot = noise.pre_forward(
                    inputs=input, targets=target_one_hot
                )

        output = self.model(input)

        if train and self.noises is not None:
            for noise in self.noises:
                output, target_one_hot = noise.post_forward(
                    outputs=output, targets=target_one_hot
                )

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.criterion(output=output, target=target_one_hot)

            l2_loss = self.l2_weight * self.l2_regulariser()
            loss += l2_loss

            if self.noises is not None:
                for noise in self.noises:
                    noise.pre_backward(loss=loss, epoch=self.epoch)

            self.backward(loss)

            if self.noises is not None:
                for noise in self.noises:
                    noise.post_backward(loss=loss, epoch=self.epoch)

            l2_gradient = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_norm_clip, norm_type=2.0
            )
            l2_weight = self.l2_norm()

            self.optimizer.step()
            self.train_metrics.update(
                output=output,
                target=target,
                loss=loss,
                l2_loss=l2_loss,
                l2_gradient=l2_gradient,
                l2_weight=l2_weight,
            )
            self.iteration += 1
        else:
            self.valid_metrics.update(output=output, target=target)

    def train(self, train_loader: torch.utils.data.DataLoader) -> None:
        """This function is responsible for the training loop."""
        self.train_metrics.reset()
        self.model.train()

        for step, batch in enumerate(train_loader):
            self._step(batch=batch, train=True)
            if step % self.printing_frequency == 0 and self.verbose:
                logging.info(
                    "##### Train step: [%03d/%03d] | %s #####",
                    len(train_loader),
                    step,
                    self.train_metrics.get_str(),
                )
            if self.args is not None and self.args.debug:
                break

    @torch.no_grad()
    def infer(self, valid_loader: torch.utils.data.DataLoader) -> None:
        """This function is responsible for the validation loop."""
        self.valid_metrics.reset()
        self.model.eval()

        for step, (input, target) in enumerate(valid_loader):
            self._step(batch=(input, target), train=False)

            if step % self.printing_frequency == 0 and self.verbose:
                logging.info(
                    "##### Valid step: [%03d/%03d] | %s #####"
                    % (len(valid_loader), step, self.valid_metrics.get_str())
                )

            if self.args is not None and self.args.debug:
                break

    def backward(self, loss: torch.Tensor, retain_graph: bool = False) -> None:
        """This function is responsible for the backward pass.

        Checks if the gradients are not NaN and if they are, it will zero them out.
        """
        if loss == loss:
            loss.backward(retain_graph=retain_graph)
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad[p.grad != p.grad] = 0

    def l2_regulariser(self) -> torch.Tensor:
        """This function is responsible for the L2 regularisation."""
        l2 = 0.0
        for p in self.model.parameters():
            if p.requires_grad:
                l2 += torch.sum(p ** 2) * 0.5
        return l2

    def l2_norm(self) -> torch.Tensor:
        """This function is responsible for the L2 as if all parameters were concatenated."""
        params = torch.cat([p.flatten() for p in self.model.parameters()])
        return params.norm(2)
