# customer-churn-ai

## good pratictes 
0. Define the Business Problem First

1. Data Layer (ML Foundations)\
  Step 1 — Get or Generate Data
    notebook experimentation\
  Step 2 — Exploratory Data Analysis (EDA)
    notebook experimentation\
  Step 3 — Data Preparation
    code inside ml/train.py\

2. Build the ML Model\
  Step 4 — Baseline Model
  Step 5 — Improve Model
  Step 6 — Save Model

3. Productionize ML with FastAPI\
  Step 7 — Create Backend Structure
  Step 8 — Load Model in FastAPI
  Step 9 — Add Feature Importance Endpoint

4. Build the Frontend Dashboard\

<br/>

## Dependencies
  - FastAPI\
    Python web framework used to expose the trained churn prediction model as a REST API.
    Handles request validation, routing, and automatic OpenAPI documentation generation.\

  - Uvicorn\
    It serves the API in development and production environments.

  - Pandas\
    Data manipulation and analysis library( data cleaning, eda )

  - Scikit-learn\
    Machine learning library for Train/validation/test splitting

  - Joblib\
    Utility library

  - Matplotlib\
    Core plotting library for data visualization, analysis charts

  - Seaborn\
    Statistical visualization library used for correlation heatmaps


<br/>
<br/>

load env : 
source /Users/frantz/Documents/customer-churn-ai/venv/bin/activate
