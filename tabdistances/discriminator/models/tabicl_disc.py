"""Small TabICL-based discriminator for table-level source classification.

Uses TabICL's ColEmbedding + RowInteraction stages as an encoder,
mean-pools across rows to get a table-level representation, and applies
a linear classification head. Trained from scratch on labelled DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from tabdistances.discriminator.base import BaseDiscriminator
from tabdistances.model.embedding import ColEmbedding
from tabdistances.model.interaction import RowInteraction
from tabdistances.model.preprocessing import TransformToNumerical

_MAX_ROWS = 512   # cap per table to keep encoding fast


class _Encoder(nn.Module):
    """ColEmbedding → RowInteraction → mean-pool → (C*E,) table vector."""

    def __init__(
        self,
        embed_dim: int,
        col_num_blocks: int,
        col_nhead: int,
        col_num_inds: int,
        row_num_blocks: int,
        row_nhead: int,
        row_num_cls: int,
    ) -> None:
        super().__init__()
        ff = embed_dim * 2

        self.col_embedder = ColEmbedding(
            embed_dim=embed_dim,
            num_blocks=col_num_blocks,
            nhead=col_nhead,
            dim_feedforward=ff,
            num_inds=col_num_inds,
            target_aware=False,
            reserve_cls_tokens=row_num_cls,
            affine=False,
        )

        self.row_interactor = RowInteraction(
            embed_dim=embed_dim,
            num_blocks=row_num_blocks,
            nhead=row_nhead,
            dim_feedforward=ff,
            num_cls=row_num_cls,
        )

        self.out_dim = row_num_cls * embed_dim

    def forward(self, X: Tensor) -> Tensor:
        """Encode a single table.

        Parameters
        ----------
        X : Tensor, shape (1, T, H)

        Returns
        -------
        Tensor, shape (1, out_dim)
        """
        T = X.shape[1]
        y_dummy = torch.zeros(1, T, device=X.device, dtype=X.dtype)
        emb = self.col_embedder(X, y_dummy, embed_with_test=True)  # (1, T, H+C, E)
        row_rep = self.row_interactor(emb)                          # (1, T, C*E)
        return row_rep.mean(dim=1)                                  # (1, C*E)


class SmallTabICLDiscriminator(BaseDiscriminator):
    """Table-level source discriminator using a small TabICL encoder.

    Stages 1+2 of the TabICL architecture (ColEmbedding + RowInteraction)
    encode each DataFrame into a fixed-size vector by mean-pooling across rows.
    A linear head classifies the resulting embedding by data source.

    The model is trained from scratch on the provided labelled DataFrames.

    Parameters
    ----------
    num_classes : int, default=2
    embed_dim : int, default=32
    col_num_blocks : int, default=1
    col_nhead : int, default=4
    col_num_inds : int, default=32
    row_num_blocks : int, default=2
    row_nhead : int, default=4
    row_num_cls : int, default=2
    lr : float, default=1e-3
    epochs : int, default=30
    batch_size : int, default=8
        Number of tables per gradient update (gradient accumulation).
    max_rows : int, default=512
        Cap on rows per table (random sample if exceeded).
    device : str or None, default=None
        Torch device string. If None, auto-selects CUDA if available, else CPU.
    """

    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 32,
        col_num_blocks: int = 1,
        col_nhead: int = 4,
        col_num_inds: int = 32,
        row_num_blocks: int = 2,
        row_nhead: int = 4,
        row_num_cls: int = 2,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 8,
        max_rows: int = _MAX_ROWS,
        device: str | None = None,
    ) -> None:
        self.num_classes   = num_classes
        self.embed_dim     = embed_dim
        self.col_num_blocks = col_num_blocks
        self.col_nhead     = col_nhead
        self.col_num_inds  = col_num_inds
        self.row_num_blocks = row_num_blocks
        self.row_nhead     = row_nhead
        self.row_num_cls   = row_num_cls
        self.lr            = lr
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.max_rows      = max_rows
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device        = torch.device(device)

        self._encoder: _Encoder | None = None
        self._head: nn.Linear | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> None:
        self._encoder = _Encoder(
            embed_dim=self.embed_dim,
            col_num_blocks=self.col_num_blocks,
            col_nhead=self.col_nhead,
            col_num_inds=self.col_num_inds,
            row_num_blocks=self.row_num_blocks,
            row_nhead=self.row_nhead,
            row_num_cls=self.row_num_cls,
        ).to(self.device)
        self._head = nn.Linear(self._encoder.out_dim, self.num_classes).to(self.device)

    def _df_to_tensor(self, df: pd.DataFrame) -> Tensor:
        """Preprocess a DataFrame to a float tensor of shape (1, T, H)."""
        if self.max_rows and len(df) > self.max_rows:
            df = df.sample(self.max_rows, random_state=42)

        # Replace inf before TransformToNumerical so SimpleImputer can handle them
        df = df.replace([np.inf, -np.inf], np.nan)

        tfm = TransformToNumerical()
        X_np = tfm.fit_transform(df).astype(np.float32)

        # Replace any remaining NaN/inf with 0
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.tensor(X_np, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _encode_table(self, df: pd.DataFrame) -> Tensor:
        """Encode one DataFrame to a (1, out_dim) table embedding."""
        X = self._df_to_tensor(df)
        return self._encoder(X)  # (1, out_dim)

    # ------------------------------------------------------------------
    # BaseDiscriminator interface
    # ------------------------------------------------------------------

    def fit(self, X: list[pd.DataFrame], y: np.ndarray) -> "SmallTabICLDiscriminator":
        self._build_model()
        self._encoder.train()
        self._head.train()

        params = list(self._encoder.parameters()) + list(self._head.parameters())
        optimiser = torch.optim.Adam(params, lr=self.lr)

        n = len(X)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)
        indices = np.arange(n)

        # Pre-split indices into batches so the inner bar steps per optimizer update
        batches = [indices[s:s + self.batch_size]
                   for s in range(0, n, self.batch_size)]
        n_batches = len(batches)

        epoch_bar = tqdm(range(self.epochs), desc="Epochs", unit="epoch")
        for epoch in epoch_bar:
            np.random.shuffle(indices)
            batches = [indices[s:s + self.batch_size]
                       for s in range(0, n, self.batch_size)]
            epoch_loss = 0.0

            batch_bar = tqdm(batches, desc=f"  Epoch {epoch+1}", unit="batch",
                             leave=False)
            for batch_idx in batch_bar:
                optimiser.zero_grad()
                batch_loss = 0.0
                for i in batch_idx:
                    emb   = self._encode_table(X[i])          # (1, out_dim)
                    logit = self._head(emb)                    # (1, num_classes)
                    loss  = F.cross_entropy(logit, y_t[i].unsqueeze(0))
                    (loss / len(batch_idx)).backward()
                    batch_loss += loss.item()
                optimiser.step()
                epoch_loss += batch_loss
                batch_bar.set_postfix(loss=f"{batch_loss / len(batch_idx):.4f}")

            batch_bar.close()
            epoch_bar.set_postfix(loss=f"{epoch_loss / n:.4f}")

        self._encoder.eval()
        self._head.eval()
        return self

    def predict(self, X: list[pd.DataFrame]) -> np.ndarray:
        if self._encoder is None or self._head is None:
            raise RuntimeError("Call fit() before predict().")
        with torch.no_grad():
            logits = torch.cat([self._head(self._encode_table(df)) for df in X], dim=0)
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: list[pd.DataFrame]) -> np.ndarray:
        if self._encoder is None or self._head is None:
            raise RuntimeError("Call fit() before predict().")
        with torch.no_grad():
            logits = torch.cat([self._head(self._encode_table(df)) for df in X], dim=0)
        return torch.softmax(logits, dim=1).cpu().numpy()
