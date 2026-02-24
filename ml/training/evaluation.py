from sklearn.metrics import (
  classification_report,
  roc_auc_score,
  confusion_matrix
)

from .threshold import find_best_threshold_f1

def evaluate_model(name,model, X_val, y_val):
  y_pred = model.predict(X_val)
  y_proba = model.predict_proba(X_val)[:, 1]

  best_threshold, best_f1 = find_best_threshold_f1(y_val, y_proba)

  print(f"Best threshold: {best_threshold}")

  print(f"Classification Report {name} model:\n")
  print(classification_report(y_val, y_pred))

  print("ROC-AUC:", roc_auc_score(y_val, y_proba))

  print("Confusion Matrix:")
  print(confusion_matrix(y_val, y_pred))
