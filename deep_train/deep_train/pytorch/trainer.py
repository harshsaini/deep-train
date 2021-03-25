import logging
import warnings
from typing import Dict
from typing import Optional

import tqdm

from ..common.constants import DISPLAY_FLOAT_PRECISION
from ..common.utils import count_trainable_parameters
from ..common.utils import get_stats
from .checkpointer import Checkpointer
from .contexts import EpochContext
from .contexts import TrainContext
from .tasks import BaseTask

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    # Ignore compat warnings from tensorboard for using deprecated packages
    # from numpy.
    import torch.utils.tensorboard

logger = logging.getLogger(__name__)


class Trainer:
    """
    This class decouples a model's definition and objective functions from it's
    training code.

    The idea is to formalize best practices around training in a decoupled
    manner so that it can be reused across different tasks without changes or
    need of prior knowledge.
    """

    def __init__(
        self,
        train_context: TrainContext,
        task_logic: BaseTask,
        checkpointer: Optional[Checkpointer] = None,
    ) -> None:
        self.train_context = train_context

        self.checkpointer = (
            checkpointer
            if checkpointer
            else Checkpointer(
                disable_checkpoints=True,
            )
        )

        self.task_logic = task_logic

    def train(self, num_epochs: int) -> None:
        """Generic training loop with optional checkpointing hooks."""
        if torch.cuda.is_available():
            logger.info("Using CUDA backend")
        else:
            logger.warning("CUDA not found, using CPU backend")

        # Tensorboard log writer
        writer = torch.utils.tensorboard.SummaryWriter(log_dir="./logs")

        # Load from checkpoint if available
        self.checkpointer.load_checkpoint(self.train_context)
        self.train_context.init_training()

        logger.info(
            "Number of trainable parameters: %s",
            count_trainable_parameters(self.train_context.model),
        )
        logger.info("Beginning training loop...")

        for epoch in range(self.checkpointer.starting_epoch, num_epochs):
            self.train_context.optimizer.zero_grad()
            epoch_context = self.task_logic.process_epoch(self.train_context)

            train_metrics = get_stats(
                epoch_context.expected.view(-1, 1).cpu().tolist(),
                epoch_context.observed.view(-1, 1).cpu().tolist(),
            )
            train_metrics["loss"] = epoch_context.loss

            Trainer.log_status(
                stage_name="training",
                epoch=epoch,
                metrics=train_metrics,
            )

            if self.train_context.dev_dataloader:
                dev_context = self.evaluate(self.train_context.dev_dataloader)

                dev_metrics = get_stats(
                    dev_context.expected.view(-1, 1).cpu().tolist(),
                    dev_context.observed.view(-1, 1).cpu().tolist(),
                )
                dev_metrics["loss"] = dev_context.loss

                Trainer.log_status(
                    stage_name="validation",
                    epoch=epoch,
                    metrics=dev_metrics,
                )

                overall_metrics = {
                    metric: {
                        "train": train_metrics[metric],
                        "dev": dev_metrics[metric],
                    }
                    for metric in train_metrics.keys() & dev_metrics.keys()
                }
            else:
                overall_metrics = {
                    metric: {"train": train_metrics[metric]}
                    for metric in train_metrics.keys()
                }

            Trainer.add_tensorboad_stats(
                epoch=epoch,
                metrics=overall_metrics,
                writer=writer,
            )

            self.checkpointer.save_checkpoint(
                current_epoch=epoch,
                current_loss=epoch_context.loss,
                train_context=self.train_context,
            )
        writer.close()

    @staticmethod
    def log_status(
        stage_name: str,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        metric_str = " ".join(
            [f"%s: %.{DISPLAY_FLOAT_PRECISION}f" for _ in range(len(metrics))],
        )
        logger.info(
            "%s metrics for epoch %s => " + metric_str,
            stage_name,
            epoch,
            *(m for k, v in metrics.items() for m in (k, v)),
        )

    @staticmethod
    def add_tensorboad_stats(
        epoch: int,
        metrics: Dict[str, Dict[str, float]],
        writer: torch.utils.tensorboard.SummaryWriter,
    ) -> None:
        for metric_name, metric_values in metrics.items():
            writer.add_scalars(
                metric_name,
                {k: v for k, v in metric_values.items()},
                epoch,
            )
        writer.flush()

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> EpochContext:
        """Runs a dataset through a model without optimizing it."""
        self.train_context.model.to(self.train_context.torch_device)
        self.train_context.model.eval()

        epoch_context = self.train_context.epoch_context_cls()
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader):
                x = batch[0].to(self.train_context.torch_device)
                y = batch[1].to(self.train_context.torch_device)

                y_hat, *extras = self.task_logic.process_minibatch(
                    x=x,
                    y=y,
                    train_context=self.train_context,
                )

                loss, expected, observed = self.task_logic.compute_loss(
                    x=x,
                    y=y,
                    y_hat=y_hat,
                    extras=extras,
                    train_context=self.train_context,
                )

                epoch_context.loss += loss.item()
                epoch_context.append_expected(expected)
                epoch_context.append_observed(observed)

        logger.debug("expected labels: %s", expected[:20])
        logger.debug("observed labels: %s", observed[:20])

        return epoch_context
