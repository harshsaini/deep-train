import logging
import os
from typing import Optional
from typing import Tuple

import pandas
import torch
import torchvision
from sklearn.model_selection import train_test_split

from deep_train.pytorch.checkpointer import Checkpointer
from deep_train.pytorch.contexts import TrainContext
from deep_train.pytorch.reproducibility import seed_pytorch
from deep_train.pytorch.tasks import SequenceClassificationTask
from deep_train.pytorch.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_LABELS = 10
LABELS = list(range(NUM_LABELS))
IMAGE_EDGE_LENGTH = 28
NUM_CHANNELS = 1
LABEL_MAPPINGS = {i: str(i) for i in range(NUM_LABELS)}


class CnnModel(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        # input channel, output channels, square convolution kernel
        # (28 +2p - 3)/1 + 1 = 26 => 26x26x3
        self.conv1 = torch.nn.Conv2d(1, 3, 3, stride=1)
        # after pooling(2,2) 26x26x3=> 13x13x3
        # (13 +2p - 3)/1 + 1 = 11 => 11x11x6
        self.conv2 = torch.nn.Conv2d(3, 6, 3, stride=1)
        # after pooling(2,2) 11x11x6=> 5x5x6
        # (5 +2p - 3)/1 + 1 = 3 => 3x3x12
        self.conv3 = torch.nn.Conv2d(6, 12, 3, stride=1)
        # an affine operation: y = Wx + b
        self.fc1 = torch.nn.Linear(108, 48)
        self.fc2 = torch.nn.Linear(48, num_classes)

    def forward(self, x: torch.Tensor) -> torch.tensor:
        # Max pooling over a (2, 2) window
        x = torch.nn.functional.max_pool2d(
            torch.nn.functional.relu(self.conv1(x)),
            2,
        )

        # If the size is a square you can only specify a single number
        x = torch.nn.functional.max_pool2d(
            torch.nn.functional.relu(self.conv2(x)),
            2,
        )
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.view(-1, 108)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.softmax(self.fc2(x), dim=-1)
        return x


class MnistTensorDataset(torch.utils.data.Dataset):
    """
    TensorDataset with support of transforms.
    """

    def __init__(
        self,
        *tensors: torch.Tensor,
        transform: Optional[torchvision.transforms.Compose] = None,
    ) -> None:
        assert all(tensors[0].size(0) == t.size(0) for t in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self) -> int:
        return self.tensors[0].size(0)


def load_data(
    path: str,
    batch_size: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    DEV_RATIO = 0.2

    data = pandas.read_csv(os.path.join(path, "mnist_train.csv"))
    logger.info("total dataset size: %s", data.shape[0])

    data_x = data.drop("label", axis=1) / 255.0
    data_y = pandas.DataFrame(
        torch.nn.functional.one_hot(
            torch.tensor(data["label"].to_numpy()),
            num_classes=NUM_LABELS,
        ).numpy(),
    )

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
        ],
    )

    train_x, dev_x, train_y, dev_y = train_test_split(
        data_x,
        data_y,
        test_size=DEV_RATIO,
    )
    logger.info("total train dataset size: %s", len(train_x))
    logger.info("total dev dataset size: %s", len(dev_x))

    def _get_dataloader(
        x: pandas.DataFrame,
        y: pandas.DataFrame,
    ) -> torch.utils.data.DataLoader:
        t_x = torch.tensor(
            x.reset_index(drop=True).values.reshape(
                len(y),
                NUM_CHANNELS,
                IMAGE_EDGE_LENGTH,
                IMAGE_EDGE_LENGTH,
            ),
        )
        t_y = torch.tensor(y.reset_index(drop=True).values).long()

        dataset = MnistTensorDataset(t_x, t_y, transform=transforms)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

    train_dataloader = _get_dataloader(train_x, train_y)
    dev_dataloader = _get_dataloader(dev_x, dev_y)

    return train_dataloader, dev_dataloader


def run(
    seed: int = 11,
    learning_rate: float = 2e-3,
    batch_size: int = 256,
    num_train_epochs: int = 3,
    path: str = "~/data/mnist",
) -> None:
    logger.info("Running the example Pytorch script")

    # Set seeds, etc for reproducibility
    seed_pytorch(seed=seed)

    # Load data
    train_dataloader, dev_dataloader = load_data(
        path=os.path.expanduser(path),
        batch_size=batch_size,
    )

    # Get model
    model = CnnModel(
        num_classes=NUM_LABELS,
    )

    # Abstracted training logic
    trainer = Trainer(
        task_logic=SequenceClassificationTask(),
        train_context=TrainContext(
            num_labels=NUM_LABELS,
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
            loss_fn=torch.nn.CrossEntropyLoss(),
            train_dataloader=train_dataloader,
            dev_dataloader=dev_dataloader,
        ),
        checkpointer=Checkpointer(disable_checkpoints=True),
    )

    trainer.train(num_epochs=num_train_epochs)


if __name__ == "__main__":
    run()
