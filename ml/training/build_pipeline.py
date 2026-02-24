from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

def build_pipeline(preprocessor):
  model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
  )

  pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model)
  ])

  return pipeline



def build_random_forest_pipeline(preprocessor):
  model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
  )

  pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model)
  ])

  return pipeline



'''

Model evaluation results :

1- build_pipeline
Classification Report:

  precision    recall  f1-score   support
0       0.91      0.72      0.81       774
1       0.52      0.81      0.63       281

    accuracy                           0.75      1055
   macro avg       0.72      0.77      0.72      1055
weighted avg       0.81      0.75      0.76      1055

ROC-AUC: 0.8531292817273121
Confusion Matrix:
[[561 213]
[ 53 228]]



Interpretation:

228 churners correctly detected
Only 53 churners missed
But 213 false positives
This model is aggressive at catching churners. Better to wrongly flag a customer than miss a churner.

2 - build_random_forest_pipeline

Classification Report:

  precision    recall  f1-score   support

0       0.82      0.90      0.86       774
1       0.64      0.47      0.54       281

    accuracy                           0.79      1055
   macro avg       0.73      0.68      0.70      1055
weighted avg       0.77      0.79      0.78      1055

ROC-AUC: 0.8198685940761584
Confusion Matrix:
[[699  75]
[150 131]]



Interpretation:

Only 131 churners detected
150 churners missed (very high)
But only 75 false positives

This model is conservative




Direct comparaison of the two models :

Recall on Churners (class 1) :
Logistic Regression: 0.81
Random Forest: 0.47

-> HUGE difference.
Logistic catches 81% of churners, while random Forest catches only 47%

Random Forest misses more than half of churners, which is dangerous for churn prediction.



The goal is to minimize missed churners (retain customers)

→ Logistic Regression is better

Because:
High recall (0.81)
High ROC-AUC
Catches most churners


Summary

Logistic Regression achieves higher ROC-AUC (0.853 vs 0.820).
Logistic Regression significantly outperforms Random Forest in recall for churners (0.81 vs 0.47).
Random Forest achieves higher overall accuracy (0.79 vs 0.75) and better precision on churn.
Given the business objective of minimizing missed churners, Logistic Regression is the preferred model.
'''