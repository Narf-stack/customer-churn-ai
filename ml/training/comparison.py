from sklearn.metrics import roc_auc_score

def compare_models(models, X_val, y_val):

  for name, model in models.items():
    y_proba = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, y_proba)
    print(f"{name} ROC-AUC: {score:.4f}")
