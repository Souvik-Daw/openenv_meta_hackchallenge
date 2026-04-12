"""Medium triage grader.

Score based on ndcg with truth.
"""

import math
import numpy as np


def dcg(scores):
    return sum(
        rel / math.log2(i + 2)  # i+2 because index starts at 0
        for i, rel in enumerate(scores)
    )

def ndcg_with_truth(predicted, truth_count):
    dcg_val = dcg(predicted)

    # Ideal list = all relevant first
    ideal = [1]*truth_count + [0]*(len(predicted)-truth_count)
    idcg_val = dcg(ideal)

    return dcg_val / idcg_val if idcg_val > 0 else 0.0

def grade(task_ranking: list, task_length: int) -> float:
    """Grade medium performance.

    Args:
        task_ranking: List of task predictions
        task_length: The length of the truth

    Returns:
        Score in [0.01, 0.99].
    """
    task_ranking = list(map(lambda x: 1 if x >= 0.99 else 0, task_ranking))
    score = ndcg_with_truth(task_ranking, task_length)
    return round(float(np.clip(score, 0.01, 0.99)), 4)


class MediumGrader:
    """Callable grader class for medium task."""

    def __call__(self, task_ranking: list, task_length: int) -> float:
        return grade(task_ranking, task_length)
