"""Numerically robust metrics for the legacy forecasting stack."""

import torch
from pytorch_forecasting.metrics import SMAPE


class StableSMAPE(SMAPE):
    """SMAPE with the mathematically bounded value at non-finite ratios.

    For finite predictions this is exactly PyTorch-Forecasting's SMAPE.  When
    an overflowing prediction produces inf/inf, the SMAPE limit is 2.  The
    legacy metric instead replaces the aggregated loss with a newly allocated
    constant, which detaches it from autograd and crashes ``backward()``.
    ``nan_to_num`` keeps the original computation graph and gives invalid
    samples zero gradient while allowing validation/early stopping to retain
    the last finite checkpoint.
    """

    def loss(self, y_pred, target):
        prediction = self.to_prediction(y_pred)
        finite_prediction = torch.isfinite(prediction)
        # Sanitize before any arithmetic. Masking only the final inf/inf ratio
        # would still leave NaN local derivatives in autograd (0 * NaN).
        safe_prediction = torch.nan_to_num(
            prediction, nan=0.0, posinf=0.0, neginf=0.0
        )
        loss = 2 * (safe_prediction - target).abs() / (
            safe_prediction.abs() + target.abs() + 1e-8
        )
        return torch.where(finite_prediction, loss, torch.full_like(loss, 2.0))
