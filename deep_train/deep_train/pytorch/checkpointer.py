import logging
import os
import random
import warnings
from typing import Any
from typing import Dict

import numpy

from ..common.constants import DISPLAY_FLOAT_PRECISION
from .contexts import TrainContext

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    # Ignore compat warnings from tensorboard for using deprecated packages
    # from numpy.
    import torch.utils.tensorboard

logger = logging.getLogger(__name__)


class Checkpointer:
    """
    Provides basic tooling for checkpointing and resuming during training.

    Checkpointing is done after successful completions of an epoch. Individual
    completions of minibatches are not tracked.
    """

    def __init__(
        self,
        checkpoint_path: str = "./checkpoints",
        disable_checkpoints: bool = False,
    ) -> None:
        # Some variables to store state and files paths
        self.disable_checkpoints = disable_checkpoints
        self.checkpoint_path = checkpoint_path
        self.state_dict_name = "state.json"
        self.state: Dict[str, Any] = {}

    def save_checkpoint(
        self,
        current_epoch: int,
        current_loss: float,
        train_context: TrainContext,
    ) -> None:
        """Save (or checkpoint) the model and training state."""
        if self.disable_checkpoints:
            logger.warning("Skipping checkpoints")
            return

        logger.debug("Trying to save a checkpoint")

        self.state = {
            # random states
            "torch_random_state": torch.get_rng_state(),
            "numpy_random_state": numpy.random.get_state(),
            "random_random_state": random.getstate(),
            # epoch info
            "checkpoint_epoch": current_epoch,
            "checkpoint_loss": current_loss,
            "resume_epoch": current_epoch + 1,  # resume from next epoch
            # model state
            "model": train_context.model.state_dict(),
            "optimizer": train_context.optimizer.state_dict(),
        }

        os.makedirs(self.checkpoint_path, exist_ok=True)
        state_file = os.path.join(self.checkpoint_path, self.state_dict_name)
        torch.save(self.state, state_file)
        logger.debug("Saved checkpoint to %s", self.checkpoint_path)

    def load_checkpoint(self, train_context: TrainContext) -> None:
        """Load model and training state from a checkpoint."""
        if self.disable_checkpoints:
            logger.warning("Skipping checkpoints")
            return

        logger.debug(
            "Trying to load saved model and training state from %s",
            self.checkpoint_path,
        )

        state_file = os.path.join(self.checkpoint_path, self.state_dict_name)

        if not os.path.exists(state_file):
            logger.info(
                "Starting fresh: could not load a checkpoint from %s",
                self.checkpoint_path,
            )
            return

        # loads model and optimizer inplace
        self.state = torch.load(state_file)
        train_context.model.load_state_dict(self.state["model"])
        train_context.optimizer.load_state_dict(self.state["optimizer"])

        # loads random state
        torch.set_rng_state(self.state["torch_random_state"])
        numpy.random.set_state(self.state["numpy_random_state"])
        random.setstate(self.state["random_random_state"])

        logger.info(
            "Successfully loaded checkpoint from epoch %s loss %s",
            self.state["checkpoint_epoch"],
            round(self.state["checkpoint_loss"], DISPLAY_FLOAT_PRECISION),
        )

    @property
    def starting_epoch(self) -> int:
        return int(self.state.get("resume_epoch", 0))
