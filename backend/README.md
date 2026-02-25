



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


