from sklearn.metrics import (
  classification_report,
  roc_auc_score,
  confusion_matrix
)

def evaluate_model(model, X_val, y_val):
  y_pred = model.predict(X_val)
  y_proba = model.predict_proba(X_val)[:, 1]

  print("Classification Report:\n")
  print(classification_report(y_val, y_pred))

  print("ROC-AUC:", roc_auc_score(y_val, y_proba))

  print("Confusion Matrix:")
  print(confusion_matrix(y_val, y_pred))
