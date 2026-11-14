"""RUL metrics used by learn_check.py (copied unchanged from the original project's metrics module).

NASA PHM scoring function (Saxena et al. 2008):
  d_i = y_pred_i - y_true_i  (positive -> over-prediction = dangerous late)
  s_i = exp(-d_i / 13) - 1   for d_i < 0  (early prediction, milder penalty)
  s_i = exp( d_i / 10) - 1   for d_i >= 0 (late  prediction, heavier penalty)
  score = sum(s_i)
Lower score is better.  A perfect predictor scores 0.
"""
from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PHM challenge asymmetric score. Lower is better; 0 is perfect.
    Over-prediction (d > 0) incurs heavier exponential penalty than under-prediction."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s))
