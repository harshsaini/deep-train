import abc
import logging
import os
import random
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy
import torch
from tqdm import tqdm

from .utils import get_perf

logger = logging.getLogger(__name__)

# TODO: Support training schedules
# https://huggingface.co/transformers/main_classes/optimizer_schedules.html#schedules
# Find similar schedules for torch
# TODO: BatchNorm?
# TODO: Fix gradient accumulation


class TorchModel:
    MAX_INPUT_LENGTH = 128

    model_name = 'model.pt'
    state_dict = 'context.json'
    optim_dict = 'optimizer.pt'

    # Add placeholders to objects need for training
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: torch.nn.modules.loss._Loss

    def __init__(self, labels: List[Any], seed: int = 0) -> None:
        self.continuing = False
        self.context: Dict[str, Any] = {}

        self.labels = labels
        self.num_labels = len(self.labels)

        # Set all seeds
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.torch_device = 'cpu'
        if torch.cuda.is_available():
            self.torch_device = 'cuda'

    @staticmethod
    def _get_dataloader(
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = True,
    ) -> torch.utils.data.DataLoader:
        'Helper to quickly create a dataloader from a dataset.'
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
        )

    @abc.abstractmethod
    def preprocess(
        self,
        *,
        filename: str,
        batch_size: int,
        max_sequence_length: int,
        cutoff: Optional[int],
        is_evaluation: bool = False,
        **kwargs: Any,
    ) -> torch.utils.data.DataLoader:
        """Create dataloaders for loading data."""

    def save_checkpoint(self, path: str) -> None:
        """Save (or checkpoint) the model and training context."""
        logger.info('Trying to save a checkpoint')
        model_file = os.path.join(path, self.model_name)
        state_file = os.path.join(path, self.state_dict)
        optimizer_file = os.path.join(path, self.optim_dict)

        torch.save(self.model.state_dict(), model_file)
        torch.save(self.optimizer.state_dict(), optimizer_file)
        torch.save(self.context, state_file)
        logger.info('Saved checkpoint to %s', path)

    def load_checkpoint(self, path: str) -> None:
        """Load model and context from a checkpoint."""
        logger.info('Trying to load saved model and context from %s', path)
        self.continuing = False

        model_file = os.path.join(path, self.model_name)
        state_file = os.path.join(path, self.state_dict)
        optimizer_file = os.path.join(path, self.optim_dict)

        if not (
            os.path.exists(model_file)
            and os.path.exists(state_file)
            and os.path.exists(optimizer_file)
        ):
            logger.warning('Could not checkpoint from %s', path)
            return

        # loads model and optimizer inplace
        self.model.load_state_dict(torch.load(model_file))
        self.optimizer.load_state_dict(torch.load(optimizer_file))
        self.context = torch.load(state_file)

        self.continuing = True
        logger.info('Successfully loaded checkpoint')

    def evaluate_batch(
        self, batch: List[torch.Tensor], **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform the basic operations to predict and compute loss for a batch.

        Assumes a batch with at least contains a sample and its corresponding
        labels. Assumes that the model returns predictions that map to labels
        specified in the constructor.
        """
        x = batch[0].to(self.torch_device)
        y = batch[1].to(self.torch_device)

        probs = self.model(x)

        # Collect predictions & compute loss
        actuals = y.reshape(-1)
        predictions = torch.argmax(probs, dim=-1).reshape(-1)
        loss = self.loss_fn(probs.view(-1, self.num_labels), actuals)
        return loss, actuals, predictions

    def validate(
        self, dataloader: torch.utils.data.DataLoader,
    ) -> Tuple[List[int], List[int]]:
        """
        Run model prediction on a dataset.

        Assumes that max sequence length and other data associated issues have
        been handled prior to adding it to the dataloader.
        """
        self.model.to(self.torch_device)
        self.model.eval()

        all_actuals: List[int] = []
        all_predictions: List[int] = []
        for batch in tqdm(dataloader, desc='Evaluating'):
            with torch.no_grad():
                loss, actuals, predictions = self.evaluate_batch(batch)
                all_actuals.extend(actuals.cpu().tolist())
                all_predictions.extend(predictions.cpu().tolist())
        return all_actuals, all_predictions

    # def predict(self, x: torch.Tensor,
    # ) -> torch.Tensor:
    #     """Evaluate the model using sample data."""
    #     if len(x.size()) < 2:
    #         raise Exception('1 dimensional vectors are not supported.')
    #     elif len(x.size()) == 2:
    #         torch.unsqueeze(torch.Tensor([1, 2, 3]), dim=0)
    #     model.to(torch_device)
    #     self.model.eval()
    #     with torch.no_grad:
    #         return self.model(x)

    def train(
        self,
        eval_file: str,
        train_file: str,
        checkpoint_dir: Optional[str] = None,
        batch_size: int = 32,
        eval_cutoff: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        inter_epoch_validation: bool = False,
        max_sequence_length: int = MAX_INPUT_LENGTH,
        num_train_epochs: int = 1000,
        output_dir: Optional[str] = None,
        patience: int = 3,
        train_cutoff: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[float, float, float]:
        """Train and evaluate a pytorch model."""
        logger.info('Using device %s', self.torch_device)

        # If checkpointing is enabled, try to load from checkpoint first
        if checkpoint_dir:
            self.load_checkpoint(path=checkpoint_dir)

        if self.continuing:
            logger.info(
                'Continuing from checkpoint at epoch %s',
                self.context['checkpoint_epoch'],
                extra={'context': self.context},
            )
        else:
            # Put whatever you want to save as part of training context
            self.context = {
                'best_f1': 0.0,
                'best_epoch': 0,
                'current_loss': 0.0,
                'checkpoint_epoch': 0,
                'epochs_since_improvement': 0,
            }
            logger.info(
                'No checkpoint found, starting fresh',
                extra={'context': self.context},
            )

        # Load and preprocess data
        # TODO: Assume the data is in a tensor and normalized, then the
        # dataloader part can be common.
        train_dataloader = self.preprocess(
            filename=train_file,
            batch_size=batch_size,
            cutoff=train_cutoff,
            max_sequence_length=max_sequence_length,
            **kwargs,
        )

        eval_dataloader = self.preprocess(
            filename=eval_file,
            batch_size=batch_size,
            cutoff=eval_cutoff,
            max_sequence_length=max_sequence_length,
            is_evaluation=True,
            **kwargs,
        )

        # Stuff we care about in the training context
        best_f1 = self.context['best_f1']
        best_epoch = self.context['best_epoch']
        current_loss = self.context['current_loss']
        checkpoint_epoch = self.context['checkpoint_epoch']
        epochs_since_improvement = self.context['epochs_since_improvement']

        # Training loop

        # TODO: Fix gradient accumulation
        # Formula needs to work off len(train_dataloader) ~ num_batches to work
        # # Ensure batch_size and gradient_accumulation_steps map correctly
        # if gradient_accumulation_steps > 1:
        #     batch_size = ((batch_size // gradient_accumulation_steps)
        #                   * gradient_accumulation_steps)

        self.model.to(self.torch_device)
        for epoch in range(checkpoint_epoch, int(num_train_epochs)):
            self.model.train()
            self.optimizer.zero_grad()
            epoch_actuals = []
            epoch_predictions = []
            current_loss = 0.0

            num_steps = 0
            num_samples = 0
            epoch_start_time = time.perf_counter()
            for step, batch in enumerate(
                tqdm(train_dataloader, desc=f'Training Epoch {epoch}'),
            ):
                loss, actuals, predictions = self.evaluate_batch(
                    batch, **kwargs,
                )

                # If we will only backprop on averaged gradients between steps
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                loss.backward()

                # Collect training metadata for reporting
                epoch_actuals.extend(actuals.cpu().tolist())
                epoch_predictions.extend(predictions.cpu().tolist())
                current_loss += loss.item()

                # Backprop iff accumulation criterion is met
                if (step + 1) % gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                num_samples += actuals.size(dim=0)
                num_steps += 1
            logger.info(
                'Processed %s samples in %s steps in %s seconds',
                num_samples,
                num_steps,
                time.perf_counter() - epoch_start_time,
            )

            # Within epoch training validation.
            # Disable `inter_epoch_validation` to improve training time.
            current_f1 = 0.0
            if inter_epoch_validation:
                logger.info('Using validation set to training compute stats')
                stage = 'InterEpochValidation'
                epoch_actuals, epoch_predictions = self.validate(
                    dataloader=eval_dataloader,
                )
            else:
                logger.info('Using training set to training compute stats')
                stage = 'InterEpochTrainingAccuracy'

            current_f1 = get_perf(
                self.labels,
                epoch_actuals,
                epoch_predictions,
                output_stage=stage,
                log=True,
            )

            if current_f1 >= best_f1:
                best_f1 = current_f1
                best_epoch = epoch
                epochs_since_improvement = 0
                logger.warning(
                    'Found new best f1 score: %s',
                    best_f1,
                    extra={'current_epoch': epoch},
                )
            else:
                epochs_since_improvement += 1
                logger.warning(
                    'Found no improvement in f1 score',
                    extra={
                        'stale_epochs': epochs_since_improvement,
                        'current_epoch': epoch,
                    },
                )

            logger.info('Total loss for epoch %s: %s', epoch, current_loss)
            logger.info('Current f1 score: %s', current_f1)
            logger.info(
                'Best f1 score: Epoch=%s Score=%s', best_epoch, best_f1,
            )

            # Stuff we care about in the training context
            self.context['best_f1'] = best_f1
            self.context['best_epoch'] = best_epoch
            self.context['current_loss'] = current_loss
            self.context['checkpoint_epoch'] = epoch + 1
            self.context['epochs_since_improvement'] = epochs_since_improvement

            # Checkpoint the model
            if checkpoint_dir:
                self.save_checkpoint(path=checkpoint_dir)

            # Early stopping criteria
            if patience and epochs_since_improvement > patience:
                logger.warning(
                    'No improvements noticed for %s epochs. Terminating.',
                    epochs_since_improvement,
                )
                break

        # Training is complete, save the trained model
        artifact_dir = output_dir or checkpoint_dir
        if artifact_dir:
            self.save_checkpoint(path=(artifact_dir))

        # validation
        actual, predicted = self.validate(dataloader=eval_dataloader)

        final_f1 = get_perf(
            self.labels,
            actual,
            predicted,
            output_stage='PostTrainValidation',
            log=True,
        )

        return final_f1, best_f1, best_epoch
