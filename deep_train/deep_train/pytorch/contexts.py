import logging
from typing import Optional
from typing import Type

import torch

logger = logging.getLogger(__name__)


class EpochContext:
    def __init__(self) -> None:
        self.loss = 0.0
        self.expected = torch.Tensor()
        self.observed = torch.Tensor()

    @staticmethod
    def _append_tensor(a: torch.Tensor, b: torch.Tensor) -> torch.tensor:
        return torch.cat([a.cpu(), b.cpu()])

    def append_expected(self, samples: torch.Tensor) -> None:
        self.expected = EpochContext._append_tensor(self.expected, samples)

    def append_observed(self, samples: torch.Tensor) -> None:
        self.observed = EpochContext._append_tensor(self.observed, samples)


class TrainContext:
    def __init__(
        self,
        num_labels: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        dev_dataloader: Optional[torch.utils.data.DataLoader] = None,
        epoch_context_cls: Type[EpochContext] = EpochContext,
    ) -> None:
        self.num_labels = num_labels
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.epoch_context_cls = epoch_context_cls

        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_device = torch.device(torch_device)

    def init_training(self) -> None:
        self.model.to(self.torch_device)
        self.model.train()
        self.optimizer.zero_grad()

    def init_eval(self) -> None:
        self.model.to(self.torch_device)
        self.model.eval()
