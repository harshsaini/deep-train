import logging
from typing import List
from typing import Optional

from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger(__name__)


def get_perf(
    labels: List[int],
    actuals: List[int],
    predictions: List[int],
    output_stage: Optional[str] = None,
    log: bool = True,
) -> float:
    """Compute the scoring metrics and optionally log the results."""
    # Individual
    precision, recall, f1, support = precision_recall_fscore_support(
        actuals, predictions, labels=list(range(len(labels))),
    )
    if log:
        for i in range(len(labels)):
            logger.info(
                'Stage: %s Label: %s => Precision: %s | Recall: %s '
                '| F1: %s | Support: %s',
                output_stage,
                labels[i],
                precision.item(i),
                recall.item(i),
                f1.item(i),
                support.item(i),
            )

    # Overall
    precision, recall, f1, support = precision_recall_fscore_support(
        actuals, predictions, average='weighted',
    )
    if log:
        logger.info(
            '%s Overall => Precision: %s | Recall: %s | F1: %s | '
            'Support: %s',
            output_stage,
            precision,
            recall,
            f1,
            support,
        )
    return f1
