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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from tqdm import tqdm

from tabdistances.discriminator.base import BaseDiscriminator
from tabdistances.model.embedding import ColEmbedding
from tabdistances.model.interaction import RowInteraction
from tabdistances.model.preprocessing import TransformToNumerical

import wandb

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
    val_frac : float, default=0.15
        Fraction of training data held out for validation loss / ROC-AUC.
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
        batch_size: int = 128,
        max_rows: int = _MAX_ROWS,
        val_frac: float = 0.15,
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
        self.val_frac      = val_frac
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

        # Per-column standardization: transformer attention is sensitive to large values
        col_mean = X_np.mean(axis=0, keepdims=True)
        col_std  = X_np.std(axis=0, keepdims=True).clip(min=1e-6)
        X_np = ((X_np - col_mean) / col_std).clip(-5.0, 5.0)

        return torch.tensor(X_np, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _encode_table(self, df: pd.DataFrame) -> Tensor:
        """Encode one DataFrame to a (1, out_dim) table embedding."""
        X = self._df_to_tensor(df)
        return self._encoder(X)  # (1, out_dim)

    def _evaluate(self, X_val: list[pd.DataFrame], y_val_t: Tensor) -> tuple[float, float]:
        """Return (val_loss, roc_auc) on a held-out set."""
        self._encoder.eval()
        self._head.eval()
        with torch.no_grad():
            logits = torch.cat([self._head(self._encode_table(df)) for df in X_val], dim=0)
        val_loss = F.cross_entropy(logits, y_val_t).item()
        proba    = torch.softmax(logits, dim=1).cpu().numpy()
        y_np     = y_val_t.cpu().numpy()
        try:
            auc = roc_auc_score(y_np, proba[:, 1] if self.num_classes == 2 else proba,
                                multi_class="ovr")
        except ValueError:
            auc = float("nan")
        self._encoder.train()
        self._head.train()
        return val_loss, auc

    # ------------------------------------------------------------------
    # BaseDiscriminator interface
    # ------------------------------------------------------------------

    def fit(self, X: list[pd.DataFrame], y: np.ndarray) -> "SmallTabICLDiscriminator":
        self._build_model()
        self._encoder.train()
        self._head.train()

        params = list(self._encoder.parameters()) + list(self._head.parameters())
        optimiser = torch.optim.Adam(params, lr=self.lr)

        # Train / val split
        tr_idx, val_idx = train_test_split(
            np.arange(len(X)), test_size=self.val_frac, stratify=y, random_state=42
        )
        X_train  = [X[i] for i in tr_idx]
        y_train  = y[tr_idx]
        X_val    = [X[i] for i in val_idx]
        y_val_t  = torch.tensor(y[val_idx], dtype=torch.long, device=self.device)

        n      = len(X_train)
        y_t    = torch.tensor(y_train, dtype=torch.long, device=self.device)
        indices = np.arange(n)

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
                    emb   = self._encode_table(X_train[i])
                    logit = self._head(emb)
                    loss  = F.cross_entropy(logit, y_t[i].unsqueeze(0))
                    (loss / len(batch_idx)).backward()
                    batch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimiser.step()
                epoch_loss += batch_loss
                batch_bar.set_postfix(loss=f"{batch_loss / len(batch_idx):.4f}")

            batch_bar.close()
            train_loss = epoch_loss / n
            val_loss, val_auc = self._evaluate(X_val, y_val_t)
            epoch_bar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                auc=f"{val_auc:.3f}",
            )
            try:
                wandb.log({"train/loss": train_loss,
                           "val/loss":   val_loss,
                           "val/roc_auc": val_auc})
            except Exception:
                pass

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
