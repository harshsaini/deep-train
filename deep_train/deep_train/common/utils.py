import logging
from typing import Dict
from typing import List

import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


logger = logging.getLogger(__name__)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_stats(
    expected: List[int],
    observed: List[int],
) -> Dict[str, float]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true=expected,
        y_pred=observed,
        zero_division=0,
        average="weighted",
    )

    accuracy = accuracy_score(y_true=expected, y_pred=observed)

    return {
        k: v or 0.0
        for k, v in zip(
            ("precision", "recall", "f1", "support", "accuracy"),
            (precision, recall, f1, support, accuracy),
        )
    }
