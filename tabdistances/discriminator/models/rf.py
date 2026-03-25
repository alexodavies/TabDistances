"""Random-forest discriminator using aggregate DataFrame features."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from tabdistances.discriminator.models.aggregate import AggregateDiscriminator


class RFDiscriminator(AggregateDiscriminator):
    """Random forest over aggregate DataFrame features.

    Parameters
    ----------
    n_estimators : int
        Number of trees (default 300).
    random_state : int
        Seed for reproducibility (default 42).
    **kwargs :
        Forwarded to :class:`sklearn.ensemble.RandomForestClassifier`.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(
            RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                **kwargs,
            )
        )
