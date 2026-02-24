
from typing import Tuple
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def apply_threshold(
  y_proba: np.ndarray,
  threshold: float = 0.5
) -> np.ndarray:
  """
  Convert predicted probabilities to binary predictions
  based on a decision threshold.

  Used everywhere we need a decision, e.g., in production inference or evaluation.
  Whenever we turn probabilities into a final prediction

  Args:
    y_proba: Predicted probabilities for positive class
    threshold: Decision cutoff

  Returns:
    Binary predictions (0 or 1)
  """
  return (y_proba >= threshold).astype(int)


def find_best_threshold_f1(
  y_true: np.ndarray,
  y_proba: np.ndarray
) -> Tuple[float, float]:
  """
  Find threshold that maximizes F1-score.
  Called during training/evaluation to select the best threshold of the given model.

  Returns:
    best_threshold, best_f1
  """
  thresholds = np.linspace(0.01, 0.99, 100)
  best_threshold = 0.5
  best_f1 = 0.0

  for t in thresholds:
    preds = apply_threshold(y_proba, t)
    score = f1_score(y_true, preds)

    if score > best_f1:
      best_f1 = score
      best_threshold = t

  return best_threshold, best_f1


def evaluate_at_threshold(
  y_true: np.ndarray,
  y_proba: np.ndarray,
  threshold: float
) -> dict:
  """
  Evaluate precision, recall and F1 at a given threshold.

  Logging metrics for a chosen threshold
  """
  preds = apply_threshold(y_proba, threshold)

  return {
    "threshold": threshold,
    "precision": precision_score(y_true, preds),
    "recall": recall_score(y_true, preds),
    "f1": f1_score(y_true, preds)
  }
