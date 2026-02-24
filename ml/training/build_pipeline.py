from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def build_pipeline(preprocessor):
  model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
  )

  pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model)
  ])

  return pipeline
