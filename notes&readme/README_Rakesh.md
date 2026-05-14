# 🚀 MLOps End-to-End POC — Loan Prediction System

> A **production-grade MLOps pipeline** implementing MLOps Maturity Level 4 — with full CI/CD/CT/CM, experiment tracking, drift monitoring, API serving, and real-time observability. Built on **MLflow + DagsHub + FastAPI + Prometheus + Grafana + Streamlit + DVC + GitHub Actions**.

---

## 📌 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Repository Structure & File Walkthrough](#3-repository-structure--file-walkthrough)
4. [Tools Used — What, Why & How](#4-tools-used--what-why--how)
5. [Integration Flow — How Everything Connects Locally](#5-integration-flow--how-everything-connects-locally)
6. [Local Setup & Run Guide](#6-local-setup--run-guide)
7. [CI/CD Pipeline Overview](#7-cicd-pipeline-overview)

---

## 1. Project Overview

This project predicts **Loan Approval** (Approved / Rejected) based on applicant data. But more importantly, it demonstrates a **complete MLOps lifecycle**:

| MLOps Pillar | Tool Used |
|---|---|
| Data Versioning | DVC + AWS S3 + DagsHub |
| Experiment Tracking | MLflow + DagsHub |
| Model Serving | FastAPI (REST API) |
| CI/CD/CT | GitHub Actions + Docker + AWS ECR |
| API Monitoring | Prometheus + Grafana |
| Drift Monitoring | Evidently AI + Streamlit |

---

## 2. Architecture Diagram

```
Raw Data (CSV)
     │
     ▼
[DVC] ──── tracks data versions ────► [DagsHub / AWS S3] (remote storage)
     │
     ▼
[training_pipeline.py] ─── trains model ──► [MLflow] logs metrics/params/model
     │                                           │
     │                                      [DagsHub] hosts MLflow UI remotely
     ▼
[trained_model.pkl] (saved locally / S3)
     │
     ▼
[main.py — FastAPI] ── exposes REST endpoints ──► /prediction_api
     │                                             /prediction_ui
     │                                             /batch_prediction
     │                                             /metrics (Prometheus)
     ▼
[Prometheus] ── scrapes /metrics every N seconds
     │
     ▼
[Grafana] ── visualizes API metrics (latency, request count, errors)
     │
     ▼
[drift_monitoring/] ── Streamlit App ── uses Evidently to detect data drift
```

---

## 3. Repository Structure & File Walkthrough

Files and folders are listed in the order they matter — from model building to serving to monitoring.

```
MLOps-E2E-POC/
│
├── prediction_model/               ← Core ML package
│   ├── config/config.py            ← All configs: paths, MLflow URI, S3 bucket, features
│   ├── processing/                 ← Data preprocessing transformers (custom sklearn pipelines)
│   ├── training_pipeline.py        ← STEP 1: Train model, log to MLflow, save artifact
│   ├── predict.py                  ← STEP 2: Load model, run single/batch inference
│   └── trained_models/             ← Stores serialized .pkl model files
│
├── drift_monitoring/               ← Evidently + Streamlit drift dashboard
│   ├── app.py                      ← Streamlit UI: loads reference + current data, renders Evidently reports
│   └── ...                         ← Reference dataset, helper scripts
│
├── tests/
│   └── test_prediction.py          ← Pytest unit tests for prediction pipeline (run in CI)
│
├── .dvc/                           ← DVC configuration (remote storage pointer)
├── .dvcignore                      ← Files DVC should ignore
├── .gitattributes                  ← Git LFS / DVC metadata handling
│
├── .github/
│   └── workflows/
│       └── main.yml                ← GitHub Actions CI/CD pipeline definition
│
├── main.py                         ← FastAPI app: all prediction endpoints + Prometheus instrumentation
├── app.py                          ← (Reserved / Streamlit entry or placeholder)
├── note.ipynb                      ← Exploratory notebook: EDA, feature engineering experiments
├── Dockerfile                      ← Containerizes the FastAPI app; runs DVC pull + training + tests on build
├── deployment.yml                  ← Kubernetes Deployment config (AWS EKS)
├── service.yml                     ← Kubernetes Service config (exposes FastAPI as LoadBalancer)
└── requirements.txt                ← All Python dependencies
```

### File-by-File Brief (in execution sequence)

---

#### `prediction_model/config/config.py`
Central configuration file. Defines:
- Feature column names (`FEATURES`)
- MLflow tracking URI (pointing to DagsHub)
- S3 bucket name and folder path for batch uploads
- Model save path

Everything downstream imports from here — single source of truth.

---

#### `prediction_model/training_pipeline.py`
The model training script. It:
1. Loads dataset from `datasets/` (versioned via DVC)
2. Applies preprocessing (imputation, encoding, scaling via sklearn Pipeline)
3. Trains a classification model (Logistic Regression / XGBoost / hyperopt-tuned)
4. Logs parameters, metrics, and the model artifact to **MLflow** (tracked on **DagsHub**)
5. Saves the trained model as a `.pkl` file to `trained_models/`

> This is the **CT (Continuous Training)** entry point — GitHub Actions triggers this when new data is pushed.

---

#### `prediction_model/predict.py`
Inference module. Exposes two functions:
- `generate_predictions(data)` — single record prediction (used by `/prediction_api` and `/prediction_ui`)
- `generate_predictions_batch(df)` — bulk prediction on a DataFrame (used by `/batch_prediction`)

Loads the saved model from `trained_models/` and applies the same preprocessing pipeline.

---

#### `note.ipynb`
Jupyter notebook used for EDA and initial experimentation. Not part of production pipeline — useful for understanding the dataset, feature distribution, and baseline model benchmarking.

---

#### `main.py` ⭐ (Core serving file)
The **FastAPI application**. This is where all tools meet:

- Imports `generate_predictions` from `prediction_model`
- Sets `mlflow.set_tracking_uri(config.TRACKING_URI)` — connects to DagsHub MLflow
- Uses `Instrumentator().instrument(app).expose(app)` from `prometheus_fastapi_instrumentator` — this automatically adds a `/metrics` endpoint that Prometheus scrapes

Endpoints exposed:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Health check |
| `/prediction_api` | POST | JSON body prediction (for programmatic access) |
| `/prediction_ui` | POST | Form parameter prediction (for UI testing) |
| `/batch_prediction` | POST | Upload CSV → returns CSV with predictions, uploads to S3 |
| `/metrics` | GET | Auto-generated Prometheus metrics endpoint |

Runs on port **8005**.

---

#### `Dockerfile`
Multi-stage container build for CI/CD:
1. Uses `python:3.10-slim-buster` base
2. Installs Python dependencies from `requirements.txt`
3. Installs `dvc[s3]`
4. Injects AWS credentials as build args (for DVC pull)
5. Runs `dvc pull --force` to fetch versioned dataset from S3
6. Runs `training_pipeline.py` to retrain the model
7. Runs `pytest` to validate predictions
8. Exposes port 8005 and starts `main.py`

> The Dockerfile is the **CI pipeline in a box** — build it and you get a fully trained, tested, and serving container.

---

#### `requirements.txt`
Key dependencies and their roles:
```
fastapi, uvicorn, gunicorn      → API serving
pydantic                        → Request validation
mlflow                          → Experiment tracking
scikit-learn, xgboost           → ML modeling
hyperopt                        → Hyperparameter tuning
prometheus_fastapi_instrumentator → Auto-expose /metrics for Prometheus
boto3 (implicit via dvc[s3])    → S3 uploads for batch results
pytest                          → CI testing
```

---

#### `drift_monitoring/app.py`
**Streamlit application** for monitoring data drift. Uses **Evidently AI** to compare:
- Reference dataset (training data distribution)
- Current dataset (recent incoming data / batch predictions)

Generates reports for:
- **Data Drift** — have feature distributions shifted?
- **Target Drift** — has the prediction distribution shifted?
- **Data Quality** — missing values, outliers, type mismatches

This is the **CM (Continuous Monitoring)** layer for data health.

---

#### `.github/workflows/main.yml`
GitHub Actions CI/CD workflow. Triggers on push to `main`:
1. Builds Docker image (passing AWS credentials as build args)
2. Inside Docker: pulls DVC data → trains model → runs tests
3. Pushes tested image to **AWS ECR**
4. Deploys to **AWS EKS** using `deployment.yml` and `service.yml`

---

#### `deployment.yml` + `service.yml`
Kubernetes manifests for production deployment on AWS EKS:
- `deployment.yml` — defines the pod spec, container image (from ECR), replica count, resource limits
- `service.yml` — exposes the deployment as a `LoadBalancer` service, making FastAPI accessible via an external IP

---

#### `.dvc/` + `.dvcignore` + `.gitattributes`
DVC configuration files. Tell DVC where the remote storage is (AWS S3 / DagsHub) and which files are tracked. The actual data files are not in Git — only `.dvc` pointer files are committed.

---

## 4. Tools Used — What, Why & How

---

### 🔬 MLflow
**What:** Open-source platform for experiment tracking, model versioning, and artifact storage.

**Why:** Without MLflow, experiments are lost — you can't compare which hyperparameters gave the best accuracy, or reproduce last week's model.

**What it does here:**
- Logs model parameters (e.g., `max_depth=5`) and metrics (e.g., `accuracy=0.82`) for every training run
- Stores the trained model as a versioned artifact
- `training_pipeline.py` calls `mlflow.log_param()`, `mlflow.log_metric()`, `mlflow.sklearn.log_model()`

**Local access:** `mlflow ui` → `http://localhost:5000`

---

### ☁️ DagsHub
**What:** A GitHub-like platform for ML — hosts both Git repos and MLflow tracking remotely.

**Why:** MLflow by default stores runs locally. DagsHub gives you a **remote MLflow server for free**, plus DVC remote storage, so your experiments are shareable and persistent across machines.

**What it does here:**
- `config.TRACKING_URI` points to `https://dagshub.com/<user>/<repo>.mlflow`
- MLflow runs are logged to DagsHub instead of localhost
- DVC remote also points to DagsHub (or S3 as an alternative)

**Integration:** Set `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` as environment variables → MLflow automatically logs to DagsHub.

---

### ⚡ FastAPI
**What:** Modern Python web framework for building high-performance REST APIs.

**Why:** Industry-standard for ML model serving. Gives automatic Swagger docs (`/docs`), Pydantic-based request validation, async support, and first-class Prometheus integration.

**What it does here:**
- Serves the trained model via 3 prediction endpoints
- Auto-validates incoming JSON using `LoanPrediction` Pydantic model
- For batch: reads uploaded CSV, runs inference, uploads result to S3, returns downloadable CSV
- Exposes `/metrics` via `prometheus_fastapi_instrumentator`

---

### 📊 Prometheus
**What:** Open-source time-series metrics collection and monitoring system.

**Why:** You need to know if your API is healthy — how many requests per second, latency, error rate. Prometheus scrapes and stores this data.

**What it does here:**
- `prometheus_fastapi_instrumentator` instruments FastAPI and adds a `/metrics` endpoint automatically
- Prometheus is configured (via `prometheus.yml`) to scrape `http://localhost:8005/metrics`
- Stores metrics like `http_requests_total`, `http_request_duration_seconds`, etc.

**Local run:** Download Prometheus binary → configure `prometheus.yml` → `./prometheus --config.file=prometheus.yml` → `http://localhost:9090`

---

### 📈 Grafana
**What:** Visualization and dashboarding tool that connects to Prometheus.

**Why:** Prometheus stores raw metrics. Grafana turns them into beautiful, actionable dashboards with graphs, alerts, and panels.

**What it does here:**
- Connects to Prometheus as a data source (`http://localhost:9090`)
- Dashboards show: API request rate, latency heatmaps, error rates, endpoint-level breakdown
- Also used to monitor Kubernetes cluster resource usage (CPU, memory per pod) in the cloud setup

**Local run:** Download Grafana → `grafana-server` → `http://localhost:3000` (admin/admin) → Add Prometheus data source → Import dashboard

---

### 🌊 Streamlit + Evidently AI
**What:** Streamlit is a Python framework for building data apps. Evidently AI generates ML monitoring reports.

**Why:** Stakeholders need to see if the model is degrading over time — not just API uptime, but whether the *data* the model sees today is different from training data.

**What it does here:**
- `drift_monitoring/app.py` is a Streamlit app
- Loads reference (training) data and current (production/batch) data
- Uses Evidently to compute drift scores and generate HTML reports embedded in the Streamlit UI
- Covers: Data Drift, Target Drift, Data Quality checks

**Local run:** `streamlit run drift_monitoring/app.py` → `http://localhost:8501`

---

### 🗃️ DVC (Data Version Control)
**What:** Git for data and models. Tracks large files (datasets, models) without storing them in Git.

**Why:** Datasets and model files are too large for Git. DVC stores them in S3/DagsHub and keeps pointer files in Git — enabling full reproducibility.

**What it does here:**
- Dataset in `prediction_model/datasets/` is tracked by DVC
- `dvc pull` in the Dockerfile fetches the correct dataset version from S3 before training
- When new data arrives → `dvc push` → commit `.dvc` file → GitHub Actions CT pipeline triggers

---

## 5. Integration Flow — How Everything Connects Locally

```
Step 1: DVC pulls data
        dvc pull
        └── fetches dataset from S3/DagsHub to prediction_model/datasets/

Step 2: Train model + log to MLflow/DagsHub
        python prediction_model/training_pipeline.py
        └── trains model
        └── mlflow logs run to DagsHub MLflow server
        └── saves .pkl to prediction_model/trained_models/

Step 3: Start FastAPI
        python main.py  (port 8005)
        └── loads trained model
        └── exposes /prediction_api, /prediction_ui, /batch_prediction
        └── exposes /metrics (Prometheus scrape target)

Step 4: Start Prometheus
        ./prometheus --config.file=prometheus.yml
        └── prometheus.yml points to http://localhost:8005/metrics
        └── scrapes FastAPI metrics every 15 seconds
        └── stores time-series data locally

Step 5: Start Grafana
        ./grafana-server
        └── open http://localhost:3000
        └── add Prometheus data source: http://localhost:9090
        └── create panels: request rate, latency, error rate

Step 6: Start Streamlit Drift Monitor
        streamlit run drift_monitoring/app.py
        └── open http://localhost:8501
        └── Evidently computes drift between reference and current data
        └── renders Data Drift, Target Drift, Data Quality reports
```

### Tool Integration Map

```
training_pipeline.py ──────────────────────────► DagsHub (MLflow runs)
         │
         ▼
trained_models/*.pkl ──► predict.py ──► main.py (FastAPI)
                                           │
                          ┌────────────────┼────────────────┐
                          ▼                ▼                 ▼
                    /prediction_api  /batch_prediction    /metrics
                                           │                 │
                                           ▼                 ▼
                                        AWS S3           Prometheus
                                                             │
                                                             ▼
                                                          Grafana

drift_monitoring/app.py ──► Evidently ──► Streamlit UI (http://localhost:8501)
```

---

## 6. Local Setup & Run Guide

### Prerequisites
- Python 3.10+
- DVC installed (`pip install dvc[s3]`)
- DagsHub account with repo connected
- AWS credentials configured (for S3 batch uploads)
- Prometheus binary ([download](https://prometheus.io/download/))
- Grafana binary ([download](https://grafana.com/grafana/download))

### Step 1: Clone & Install
```bash
git clone https://github.com/Rkcpr011/MLOps-E2E-POC-i-mubahsir-hasan.git
cd MLOps-E2E-POC-i-mubahsir-hasan
pip install -r requirements.txt
pip install dvc[s3]
```

### Step 2: Configure Environment Variables
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
export MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
export AWS_ACCESS_KEY_ID=<your-key>
export AWS_SECRET_ACCESS_KEY=<your-secret>
```

### Step 3: Pull Data & Train Model
```bash
dvc pull                                              # fetch dataset from remote
python prediction_model/training_pipeline.py          # train + log to MLflow/DagsHub
```

### Step 4: Start FastAPI
```bash
python main.py
# API:          http://localhost:8005
# Swagger docs: http://localhost:8005/docs
# Metrics:      http://localhost:8005/metrics
```

### Step 5: Configure & Start Prometheus
Edit `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'fastapi'
    static_configs:
      - targets: ['localhost:8005']
```
```bash
./prometheus --config.file=prometheus.yml
# Prometheus UI: http://localhost:9090
```

### Step 6: Start Grafana
```bash
./bin/grafana-server
# Grafana UI: http://localhost:3000  (admin / admin)
# Add data source: Prometheus → http://localhost:9090
```

### Step 7: Start Drift Monitoring
```bash
streamlit run drift_monitoring/app.py
# Drift Dashboard: http://localhost:8501
```

### Step 8: Run Tests
```bash
pytest -v tests/test_prediction.py
```

### All Services at a Glance

| Service | URL | Purpose |
|---|---|---|
| FastAPI | http://localhost:8005 | Model serving |
| FastAPI Docs | http://localhost:8005/docs | Interactive API docs |
| FastAPI Metrics | http://localhost:8005/metrics | Prometheus scrape target |
| MLflow (local) | http://localhost:5000 | Experiment tracking (if local mode) |
| DagsHub MLflow | https://dagshub.com/... | Remote experiment tracking |
| Prometheus | http://localhost:9090 | Metrics storage |
| Grafana | http://localhost:3000 | Metrics visualization |
| Streamlit | http://localhost:8501 | Drift monitoring dashboard |

---

## 7. CI/CD Pipeline Overview

Triggered on every push to `main` branch via **GitHub Actions**:

```
Push to main
    │
    ▼
GitHub Actions (main.yml)
    │
    ├── Build Docker Image
    │       └── dvc pull        (fetch latest data from S3)
    │       └── training_pipeline.py  (retrain model)
    │       └── pytest          (validate predictions)
    │
    ├── Push Docker Image ──► AWS ECR
    │
    └── Deploy ──► AWS EKS
            └── kubectl apply -f deployment.yml
            └── kubectl apply -f service.yml
            └── FastAPI live on LoadBalancer external IP
```

> **CT (Continuous Training):** When new data is pushed to DVC remote and the `.dvc` pointer file is committed to Git, GitHub Actions triggers a full retrain — automatically.

---

## 🙏 Acknowledgements

Project inspired by and adapted from the original work of **Mubahsir Hasan** as a learning and portfolio reference implementation of MLOps best practices at maturity level 4.

---

## 📂 Tech Stack Summary

`Python` · `FastAPI` · `MLflow` · `DagsHub` · `DVC` · `Evidently AI` · `Streamlit` · `Prometheus` · `Grafana` · `Docker` · `GitHub Actions` · `AWS S3` · `AWS ECR` · `AWS EKS` · `scikit-learn` · `XGBoost` · `Hyperopt` · `Pytest`
