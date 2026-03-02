



## Architectural tree
```bash
customer-churn-ai/
│
├── backend/ ( inference logic )
│   ├── app/
│   │   ├── main.py
│   │   ├── api/ ( routing layer )
│   │   │   └── v1/
│   │   │       └── endpoints/
│   │   │           └── predict.py
│   │   ├── core/
│   │   │   └── config.py
│   │   ├── schemas/ ( Pydantic request/response models )
│   │   │   └── churn.py
│   │   ├── services/ ( business logic )
│   │   │   └── prediction_service.py
│   │   └── ml/
│   │       ├── model_loader.py
│   │       └── pipeline.py
│   │
│   └── pyproject.toml
│
├── ml/   (training logic/ code ) 
│
└── models/ ( serialized artifacts )
    ├── model.pkl
    ├── preprocessor.pkl
    └── threshold.json
``` 


## Set up

```bash
poetry new backend
cd backend
poetry add fastapi uvicorn scikit-learn joblib pydantic
``` 






## What is lifespan?

lifespan is a hook for:
> - Startup tasks
> - Shutdown tasks

We initialise it this way:
```bash
app = FastAPI(lifespan=lifespan)
``` 

### Why We Use It for ML

Loading a model:
> - Is expensive
> - Should happen once
> - Should not happen per request

So lifespan ensures:

```bash
App starts
↓
Model loads
↓
App serves requests
↓
App shuts down
``` 

## What is asynccontextmanager in main.py?

asynccontextmanager is a Python utility that lets us define startup and shutdown logic around an application lifecycle.

It works like this:

```bash
@asynccontextmanager
async def lifespan(app: FastAPI):
  # code BEFORE yield → runs at startup
  yield
  # code AFTER yield → runs at shutdown
``` 

In this app, when FastAPI starts → load model.
When FastAPI shuts down → (nothing here yet)




# GET /feature-importance endpoint

It shows : 
  > understanding of model interpretability
  > can expose ML insights via API

We are using the logistic regression model where:
  > Positive coefficient → increases probability of churn
  > Negative coefficient → decreases probability of churn
  > Larger absolute value → stronger impact



# Normal Loop VS List comprehension loop

1 -  Normal Loop

```bash
importance = []

for feature, coef in zip(feature_names, coefficients):
  importance.append({
    "feature": feature,
    "importance": float(coef)
  })
``` 

- More readable
- Easier to debug
- Better for complex logic


2 - List Comprehension Version

```bash
importance = [
  {
    "feature": feature,
    "importance": float(coef)
  }
  for feature, coef in zip(feature_names, coefficients)
]
``` 

- More concise
- Expresses intent clearly
- Slightly faster

## When To Use Which?
- List Comprehension :
  - simple transformation
  - No complex branching logic
  - No debugging needed
  - fits comfortably on 1–3 lines

- Normal Loop :
  - Logic is complex
  - need conditionals
  - need debugging
  - need multiple statements

### Difference Conceptually

List comprehension is for:

“Create a new list from another iterable.”

A normal loop is for:

“Execute procedural logic step-by-step.”

One is declarative, the other is imperative.



# Pipeline


A Pipeline is an automated ML workflow. It is the entire machine.

It chains steps like:

```bash
Raw Data
   ↓
Imputer (fill missing values)
   ↓
Scaler (normalize numbers)
   ↓
OneHotEncoder (convert categories to numbers)
   ↓
LogisticRegression
   ↓
Prediction
``` 

# Classifier
The mathematical model used for the project, the brain. Ex, the "LogisticRegression"

It only knows how to:
> Receive numbers
> Apply weights
> Compute probability
> Output prediction



# OneHotEncoder 

It converts part of the data, like this:
```bash
Contract = Two year
``` 

Into:
```bash
Contract_Month-to-month = 0
Contract_One year = 0
Contract_Two year = 1
``` 


It creates new columns for each category.
Converts categories → numeric columns

# StandardScaler

Current numeric features are as such :

```bash
tenure → 1 to 72
MonthlyCharges → 20 to 120
``` 
There are having different scales.
Logistic regression works better when features are centered and scaled.

StandardScaler will applies so math so the values have a :

```bash
Mean = 0
Std = 1
``` 

This prevents one feature from dominating the prediction due to large magnitude.


# Metadata 

```bash
Train model
Evaluate model
Compute metrics
Create metadata
Save model
Save metadata
``` 


# Endpoints 
Production ML APIs usually have:

```bash
/predict
/model/metadata
/model/health
/model/feature-importance
/model/explain   (SHAP)
/metrics         (Prometheus monitoring)
``` 
