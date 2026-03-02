import sklearn
import platform
from datetime import datetime, timezone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



def build_metadata(model, X_train, y_train, threshold):
  now = datetime.now(timezone.utc)

  y_pred = model.predict(X_train)
  y_prob = model.predict_proba(X_train)[:, 1]

  metadata = {
    "model_name": "Customer Churn Predictor",
    "model_type": type(model.named_steps["classifier"]).__name__,
    "version": "1.0.0",
    "training_date": now.isoformat(),
    "training_samples": len(X_train),
    "features_count": len(
      model.named_steps["preprocessor"].get_feature_names_out()
    ),
    "class_labels": model.named_steps["classifier"].classes_.tolist(),
    "threshold": threshold,
    "metrics": {
      "accuracy": accuracy_score(y_train, y_pred),
      "precision": precision_score(y_train, y_pred),
      "recall": recall_score(y_train, y_pred),
      "f1_score": f1_score(y_train, y_pred),
      "roc_auc": roc_auc_score(y_train, y_prob)
    },
    "environment": {
      "python_version": platform.python_version(),
      "sklearn_version": sklearn.__version__
    }
  }

  return metadata