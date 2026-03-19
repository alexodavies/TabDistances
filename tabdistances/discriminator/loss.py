"""Combined quality score for synthetic data evaluation.

Euclidean distance from the origin toward the ideal (1, 1, 1) quality point,
normalised to [0, 1].
"""

from __future__ import annotations

import math


def combined_score(
    recall: float,
    precision: float,
    discriminator_auc: float,
) -> float:
    """Euclidean quality score for synthetic data, normalised to [0, 1].

    All three components are quality scores in [0, 1] (higher = better):

      recall      = fraction of B's (real) points whose nearest neighbour in A
                    (synth) falls within A's 95th-percentile k-NN radius.

      precision   = fraction of A's (synth) points whose nearest neighbour in B
                    (real) falls within B's 95th-percentile k-NN radius.

      auc_quality = (1 - discriminator_auc) * 2
                    Realism: normalised from [0.5, 1]  $\\in$  [1, 0].
                    AUC = 0.5 (indistinguishable, ideal)  $\\in$  auc_quality = 1.0
                    AUC = 1.0 (easily separated, worst)   $\\in$  auc_quality = 0.0

    Returns
    -------
    float
        Score in [0, 1].  MAXIMISE toward 1.0.
        score = sqrt(recall^2 + precision^2 + auc_quality^2) / sqrt(3)
    """
    _NORM = math.sqrt(3.0)
    auc_quality = (1.0 - discriminator_auc) * 2.0
    return math.sqrt(
        recall ** 2
        + precision ** 2
        + auc_quality ** 2
    ) / _NORM
