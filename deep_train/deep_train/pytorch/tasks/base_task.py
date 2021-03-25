import abc
import logging
from typing import List
from typing import Tuple

import torch
import tqdm

from ..contexts import EpochContext
from ..contexts import TrainContext

logger = logging.getLogger(__name__)


class BaseTask:
    """
    This class attempts to decouple task specific logic during training from a
    model's infrastructure necessary for training.

    Note: This sample illustrates a simple use case that works well for most
    tasks. However, it must be noted that for more involved tasks, this class
    can be extended to cater for more diverse objectives.
    """

    @abc.abstractmethod
    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        extras: List[torch.Tensor],
        train_context: TrainContext,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss and return loss, expectations and predictions."""

    def process_minibatch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        train_context: TrainContext,
    ) -> List[torch.Tensor]:
        """
        Abstracts the minibatch processing logic.

        Expects the first element of the output(s) to be the predictions.
        """
        output = train_context.model(x)
        if isinstance(output, torch.Tensor):
            return [output]
        return output

    def process_epoch(self, train_context: TrainContext) -> EpochContext:
        epoch_context = train_context.epoch_context_cls()

        for batch in tqdm.tqdm(train_context.train_dataloader):
            # In several use cases, a mini-batch evaluation logic is
            # coupled tightly to the particular model and loss function.
            # We should actually decouple the processing logic within this
            # loop (called the mini-batch processing logic from now
            # onwards) so that users can provide their own custimizations
            # if necessary or use the stock implementations provided.
            x = batch[0].to(train_context.torch_device)
            y = batch[1].to(train_context.torch_device)

            y_hat, *extras = self.process_minibatch(
                x=x,
                y=y,
                train_context=train_context,
            )

            step_loss, expected, observed = self.compute_loss(
                x=x,
                y=y,
                y_hat=y_hat,
                extras=extras,
                train_context=train_context,
            )

            step_loss.backward()

            train_context.optimizer.step()
            train_context.optimizer.zero_grad()

            epoch_context.loss += step_loss.item()
            epoch_context.append_expected(expected)
            epoch_context.append_observed(observed)
        return epoch_context
