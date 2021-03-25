import random

import numpy
import torch


def seed_pytorch(seed: int) -> None:
    """Seeds various random generators to help in reproducibility."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # TODO: Enabling this makes CNN operations deterministic but at a lowered
    # speed. Disabling it in favour of speed.
    # torch.set_deterministic(True)
