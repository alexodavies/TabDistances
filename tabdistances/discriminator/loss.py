"""Combined quality score for synthetic data evaluation.

Euclidean distance from the origin toward the ideal (1, 1, 1) quality point,
normalised to [0, 1].
"""

from __future__ import annotations

import math


def combined_score(
    coverage_real_by_synth: float,
    coverage_synth_by_real: float,
    discriminator_auc: float,
) -> float:
    """Euclidean quality score for synthetic data, normalised to [0, 1].

    All three components are quality scores in [0, 1] (higher = better):

      cov_fwd     = coverage_real_by_synth
                    Recall: what fraction of real data is covered by synthetic.

      cov_rev     = coverage_synth_by_real
                    Precision: what fraction of synthetic is close to real.

      auc_quality = (1 - discriminator_auc) * 2
                    Realism: normalised from [0.5, 1] → [1, 0].
                    AUC = 0.5 (indistinguishable, ideal) → auc_quality = 1.0
                    AUC = 1.0 (easily separated, worst)  → auc_quality = 0.0

    Returns
    -------
    float
        Score in [0, 1].  MAXIMISE toward 1.0.
        score = sqrt(cov_fwd^2 + cov_rev^2 + auc_quality^2) / sqrt(3)
    """
    _NORM = math.sqrt(3.0)
    auc_quality = (1.0 - discriminator_auc) * 2.0
    return math.sqrt(
        coverage_real_by_synth ** 2
        + coverage_synth_by_real ** 2
        + auc_quality ** 2
    ) / _NORM
