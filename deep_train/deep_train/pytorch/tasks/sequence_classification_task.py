from typing import List
from typing import Tuple

import torch

from ..contexts import TrainContext
from .base_task import BaseTask


class SequenceClassificationTask(BaseTask):
    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        extras: List[torch.Tensor],
        train_context: TrainContext,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss for assigning single class labels to an input sample.

        Returns loss, expected labels and observed (predicted) labels.
        """
        expected = torch.argmax(y, dim=-1)
        observed = y_hat.view(-1, train_context.num_labels)

        loss = train_context.loss_fn(observed, expected)

        return loss, expected, torch.argmax(observed, dim=-1)
