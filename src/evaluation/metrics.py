from typing import Optional, Dict, Union, Any, List, Callable
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchmetrics


class Error(torchmetrics.Metric):
    """Calculate the error for classification."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model.
            target (torch.Tensor): Ground truth values.
        """
        preds = torch.argmax(preds, dim=1)
        self.sum += torch.sum(preds != target)
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the error."""
        return self.sum / self.count


class ClassificationNegativeLogLikelihood(torchmetrics.Metric):
    """Calculate the negative log-likelihood for classification."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model.
            target (torch.Tensor): Ground truth values.
        """
        one_hot = F.one_hot(target, num_classes=preds.shape[1]).float()
        self.sum += torch.sum(-one_hot * torch.log(preds + 1e-8))
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the negative log-likelihood."""
        return self.sum / self.count


class BrierScore(torchmetrics.Metric):
    """Calculate the Brier score for classification."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model.
            target (torch.Tensor): Ground truth values.
        """
        one_hot = F.one_hot(target, num_classes=preds.shape[1]).float()
        self.sum += torch.sum((preds - one_hot) ** 2)
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the Brier score."""
        return self.sum / self.count


class PredictiveEntropy(torchmetrics.Metric):
    """Calculate the predictive entropy for classification."""

    is_differentiable = False
    higher_is_better = True
    full_state_update = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model.
            target (torch.Tensor): Ground truth values.
        """
        self.sum += torch.sum(-preds * torch.log(preds + 1e-8))
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the predictive entropy."""
        return self.sum / self.count


class RegressionNegativeLogLikelihood(torchmetrics.Metric):
    """Calculate the negative log-likelihood for regression.

    The negative log-likelihood is a proper metric for evaluating the uncertainty
    in regression. It measures the mean log-likelihood of the true outcome.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model, mean and or variance.
                                    Mean is assumed to be the first element. Variance
                                    is assumed to be the second element.
            target (torch.Tensor): Ground truth values.
        """
        mean = preds[:, 0]
        variance = preds[:, 1] if preds.shape[1] > 1 else torch.ones_like(mean)
        # Clamp the variance to avoid numerical issues
        variance = torch.clamp(variance, min=1e-8, max=1e8)
        self.sum += F.gaussian_nll_loss(
            mean.squeeze(), target.squeeze(), variance.squeeze(), reduction="sum"
        )
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the negative log-likelihood."""
        return self.sum / self.count


class MeanSquaredError(torchmetrics.Metric):
    """Calculate the mean squared error for regression."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model, mean and or variance.
                                    Mean is assumed to be the first element. Variance
                                    is assumed to be the second element.
            target (torch.Tensor): Ground truth values.
        """
        mean = preds[:, 0]
        self.sum += F.mse_loss(mean.squeeze(),
                               target.squeeze(), reduction="sum")
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the mean squared error."""
        return self.sum / self.count


class RootMeanSquaredError(MeanSquaredError):
    """Calculate the root mean squared error for regression."""

    def compute(self) -> torch.Tensor:
        """Compute the root mean squared error."""
        return torch.sqrt(self.sum / self.count)


class MeanAbsoluteError(torchmetrics.Metric):
    """Calculate the mean absolute error for regression."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model, mean and or variance.
                                    Mean is assumed to be the first element. Variance
                                    is assumed to be the second element.
            target (torch.Tensor): Ground truth values.
        """
        mean = preds[:, 0]
        self.sum += F.l1_loss(mean.squeeze(),
                              target.squeeze(), reduction="sum")
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the mean absolute error."""
        return self.sum / self.count


METRICS_MAPPING = {
    "error": "Error [\%]",
    "ece": "Expected Calibration Error [\%]",
    "entropy": "Entropy",
    "brier": "Brier Score",
    "nll": "NLL [nats]",
    "mse": "Mean Squared Error",
    "rmse": "Root Mean Squared Error",
    "mae": "Mean Absolute Error",
    "loss": "Total Loss",
    "l2_loss": "L2 Loss",
    "l2_weight": "L2 Weight Norm",
    "l2_gradient": "L2 Gradient Norm",
    "l2_weight_complete": "L2 Weight Norm",
    "l2_gradient_complete": "L2 Gradient Norm",
    "f1": "F1 Score [0-1]",
    "auroc": "AUROC [0-1]",
}

DATASET_MAPPING = {
    "train": "Train",
    "valid": "Valid",
    "test": "Test",
    "augmentations": "Aug.",
}

METRICS_DESIRED_TENDENCY_MAPPING = {
    "error": "down",
    "ece": "down",
    "entropy": "up",
    "brier": "down",
    "nll": "down",
    "mse": "down",
    "rmse": "down",
    "mae": "down",
    "loss": "down",
    "f1": "up",
    "auroc": "up",
    "l2_loss": "down",
    "l2_weight": "down",
    "l2_gradient": "down",
    "l2_weight_complete": "down",
    "l2_gradient_complete": "down",
}


class Metric:
    """This is a general metric class that can be used for any task. It is not specific to classification or regression.

    Args:
        output_size (int): The size of the output of the model.
        writer (SummaryWriter): Tensorboard writer.
        log (Dict[str, Any]): Dictionary to log the results.
    """

    metric_labels = ["loss", "l2_loss", "l2_weight", "l2_gradient"]

    def __init__(
        self,
        output_size: int,
        writer: Optional[SummaryWriter] = None,
        log: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.writer = writer
        self.log = log
        self.output_size = output_size

        self.loss = AverageMeter()
        self.l2_loss = AverageMeter()

        self.l2_weight = AverageMeter()
        self.l2_gradient = AverageMeter()

        self.l2_weight_complete = CompleteMeter()
        self.l2_gradient_complete = CompleteMeter()

        self.metrics = [self.loss, self.l2_loss,
                        self.l2_weight, self.l2_gradient]

    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics:
            if hasattr(metric, "reset"):
                metric.reset()
            else:
                raise ValueError(
                    f"Metric {metric} does not have a reset method.")

        self.l2_weight_complete.reset()
        self.l2_gradient_complete.reset()

    def get_metric_value(self, metric: Union[Callable, torchmetrics.Metric]) -> float:
        """Get the value of a metric."""
        val = None
        if hasattr(metric, "avg"):
            val = metric.avg
        elif hasattr(metric, "compute"):
            val = metric.compute()
        else:
            val = metric()
        return val if isinstance(val, float) else val.item()

    def scalar_logging(self, info: str, iteration: int) -> None:
        """Log all metrics to tensorboard if `SummaryWriter` is provided."""
        if self.writer is None:
            return
        for i, metric in enumerate(self.metrics):
            val = self.get_metric_value(metric)
            self.writer.add_scalar(
                info + "/" +
                METRICS_MAPPING[self.metric_labels[i]], val, iteration
            )

        if self.log is not None:
            for i, metric in enumerate(self.metrics):
                val = self.get_metric_value(metric)
                if info not in self.log:
                    self.log[info] = {}
                if iteration not in self.log[info]:
                    self.log[info][iteration] = {}
                self.log[info][iteration][self.metric_labels[i]] = val

            # Add also the complete log of l2_weight and l2_gradient
            if info not in self.log:
                self.log[info] = {}
            if iteration not in self.log[info]:
                self.log[info][iteration] = {}
            self.log[info][iteration][
                "l2_weight_complete"
            ] = self.l2_weight_complete.get_complete()
            self.log[info][iteration][
                "l2_gradient_complete"
            ] = self.l2_gradient_complete.get_complete()

    def get_str(self) -> str:
        """Get a string representation of all metrics."""
        s = ""
        for i, metric in enumerate(self.metrics):
            val = self.get_metric_value(metric)
            s += f"{METRICS_MAPPING[self.metric_labels[i]]}: {str(val)} "
        return s

    def get_packed(self) -> Dict[str, float]:
        """Get a dictionary of all metrics."""
        d = {}
        for i, metric in enumerate(self.metrics):
            val = self.get_metric_value(metric)
            d[self.metric_labels[i].lower()] = val
        return d

    def update(
        self,
        loss: Optional[Union[float, torch.Tensor]] = None,
        l2_loss: Optional[Union[float, torch.Tensor]] = None,
        l2_weight: Optional[Union[float, torch.Tensor]] = None,
        l2_gradient: Optional[Union[float, torch.Tensor]] = None,
    ) -> None:
        """Update all metrics.

        Args:
            loss (float, optional): Loss. Defaults to 0.0.
            l2_loss (float, optional): L2 loss. Defaults to 0.0.
            l2_weight (float, optional): L2 weight norm. Defaults to 0.0.
            l2_gradient (float, optional): L2 gradient norm. Defaults to 0.0.
        """
        for metric, container in zip(
            [loss, l2_loss, l2_weight, l2_gradient],
            [self.loss, self.l2_loss, self.l2_weight, self.l2_gradient],
        ):
            if metric is not None:
                metric = metric if not hasattr(
                    metric, "item") else metric.item()
                container.update(metric, 1)

        if l2_weight is not None:
            self.l2_weight_complete.update(l2_weight.item(), 1)

        if l2_gradient is not None:
            self.l2_gradient_complete.update(l2_gradient.item(), 1)


class ClassificationMetric(Metric):
    """This is a metric class for classification tasks.

    Args:
        output_size (int): Number of output classes.
        writer (SummaryWriter): Tensorboard writer.
        log (Dict[str, Any]): Dictionary to log the results.
    """

    metric_labels = [
        "loss",
        "l2_loss",
        "l2_weight",
        "l2_gradient",
        "nll",
        "error",
        "entropy",
        "brier",
        "ece",
        "f1",
        "auroc",
    ]

    def __init__(
        self,
        output_size: int,
        writer: Optional[SummaryWriter] = None,
        log: Optional[Dict[str, Any]] = None,
    ) -> None:
        super(ClassificationMetric, self).__init__(output_size, writer, log)
        self.entropy = PredictiveEntropy()
        self.ece = torchmetrics.CalibrationError(
            n_bins=10, task="multiclass", norm="l1", num_classes=output_size
        )
        self.nll = ClassificationNegativeLogLikelihood()
        self.brier = BrierScore()
        self.error = Error()
        self.f1 = torchmetrics.F1Score(
            num_classes=output_size, task="multiclass", average="macro"
        )
        self.auroc = torchmetrics.AUROC(
            num_classes=output_size, task="multiclass", average="macro"
        )

        self.metrics += [
            self.nll,
            self.error,
            self.entropy,
            self.brier,
            self.ece,
            self.f1,
            self.auroc,
        ]

    @torch.no_grad()
    def update(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Union[float, torch.Tensor],
    ) -> None:
        """Update all metrics.

        The output has to be a tensor of shape `(batch_size, output_size)`.

        Args:
            output (torch.Tensor): Model output.
            target (torch.Tensor): Target.
        """
        super(ClassificationMetric, self).update(**kwargs)
        output = output.detach()
        output = F.softmax(output, dim=1)

        # Check that the metrics are in the right device
        if not self.nll.device == output.device:
            self.nll.to(output.device)
            self.error.to(output.device)
            self.entropy.to(output.device)
            self.brier.to(output.device)
            self.ece.to(output.device)
            self.f1.to(output.device)
            self.auroc.to(output.device)

        self.nll.update(output, target)
        self.error.update(output, target)
        self.entropy.update(output, target)
        self.brier.update(output, target)
        self.ece.update(output, target)
        self.f1.update(output, target)
        self.auroc.update(output, target)


class RegressionMetric(Metric):
    """This is a metric class for regression tasks.

    Args:
        output_size (int): Output size. Not used.
        writer (SummaryWriter): Tensorboard writer.
        log (Dict[str, Any]): Dictionary to log the results.
    """

    metric_labels = [
        "loss",
        "l2_loss",
        "l2_weight",
        "l2_gradient",
        "nll",
        "rmse",
        "mse",
        "mae",
    ]

    def __init__(
        self,
        output_size: int,
        writer: Optional[SummaryWriter] = None,
        log: Optional[Dict[str, Any]] = None,
    ) -> None:
        super(RegressionMetric, self).__init__(
            output_size=output_size, writer=writer, log=log
        )

        self.rmse = RootMeanSquaredError()
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.nll = RegressionNegativeLogLikelihood()

        self.metrics += [self.nll, self.rmse, self.mse, self.mae]

    @torch.no_grad()
    def update(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Union[float, torch.Tensor],
    ) -> None:
        """Update all metrics.

        The output has to be a tensor of shape `(batch_size, 2)`.

        The first column is the mean and the second column is the variance.

        Args:
            output (torch.Tensor): Model output.
            target (torch.Tensor): Target.
        """
        super(RegressionMetric, self).update(**kwargs)

        output = output.detach()
        mean = output[:, 0]
        # var = torch.exp(output[:, 1])
        var = torch.clamp(torch.exp(output[:, 1]), min=1e-4, max=1e4)
        output = torch.cat([mean.unsqueeze(1), var.unsqueeze(1)], dim=1)

        # Check that the metrics are in the right device
        if not self.nll.device == output.device:
            self.nll.to(output.device)
            self.rmse.to(output.device)
            self.mse.to(output.device)
            self.mae.to(output.device)

        self.nll.update(output, target)
        self.rmse.update(output, target)
        self.mse.update(output, target)
        self.mae.update(output, target)


def metric_factory(
    task: str,
    output_size: int,
    writer: Optional[SummaryWriter] = None,
    log: Optional[Dict[str, Any]] = None,
) -> Metric:
    """This is a metric factory.

    Args:
        task (str): Task name.
        output_size (int): Output size.
        writer (Optional[SummaryWriter], optional): Tensorboard writer. Defaults to None.

    Returns:
        Metric: Metric class.
    """
    if task == "classification":
        return ClassificationMetric(output_size, writer, log)
    elif task == "regression":
        return RegressionMetric(output_size, writer, log)
    else:
        raise ValueError(f"Task {task} is not supported.")


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class CompleteMeter(AverageMeter):
    """Computes and stores the average and current value."""

    def __init__(self) -> None:
        super(CompleteMeter, self).__init__()
        self.complete = []

    def reset(self) -> None:
        super(CompleteMeter, self).reset()
        self.complete = []

    def update(self, val: float, n: int = 1) -> None:
        super(CompleteMeter, self).update(val, n)
        self.complete.append(val.item() if hasattr(val, "item") else val)

    def get_complete(self) -> List[Any]:
        return self.complete
