



## Architecture tree

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


## Set up

```bash
poetry new backend
cd backend
poetry add fastapi uvicorn scikit-learn joblib pydantic
``` 
