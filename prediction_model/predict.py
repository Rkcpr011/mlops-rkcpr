import pandas as pd
import numpy as np
from prediction_model.config import config  
import mlflow
import os
import joblib
# Local cache path
MODEL_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "prediction_model", 
    "trained_models", 
    "cached_model.pkl"
)

def load_best_model():
    # Agar local cache hai toh seedha load karo
    if os.path.exists(MODEL_CACHE_PATH):
        print("Loading model from local cache...")
        return joblib.load(MODEL_CACHE_PATH)
    
    # Nahi hai toh DagsHub se download karo aur cache karo
    print("Downloading model from DagsHub...")
    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id
    runs_df = mlflow.search_runs(
        experiment_ids=experiment_id,
        order_by=['metrics.f1_score DESC']
    )
    best_run_id = runs_df.iloc[0]['run_id']
    best_model_uri = f'runs:/{best_run_id}/{config.MODEL_NAME}'
    
    model = mlflow.sklearn.load_model(best_model_uri)
    
    # Local cache mein save karo
    os.makedirs(os.path.dirname(MODEL_CACHE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_CACHE_PATH)
    print(f"Model cached at: {MODEL_CACHE_PATH}")
    return model


def generate_predictions(data_input):
    data = pd.DataFrame(data_input)
    experiment_name = config.EXPERIMENT_NAME
    experiment = mlflow.get_experiment_by_name(experiment_name)

       # Debug — ye print karo
    print(f"Experiment: {experiment}")
    print(f"Experiment ID: {experiment.experiment_id}")

    experiment_id = experiment.experiment_id
    runs_df=mlflow.search_runs(experiment_ids=experiment_id,order_by=['metrics.f1_score DESC'])

     # Debug — runs dekho
    print(f"Total runs: {len(runs_df)}")
    print(f"Runs columns: {runs_df.columns.tolist()}")
    print(f"First run:\n{runs_df.iloc[0]}")

    best_run=runs_df.iloc[0]
    best_run_id=best_run['run_id']

     # Debug
    print(f"Best run ID: {best_run_id}")
    print(f"Model name: {config.MODEL_NAME}")

    best_model = f'runs:/{best_run_id}/{config.MODEL_NAME}'
    # debug
    print(f"Model URI: {best_model}")

    # loan_prediction_model=mlflow.sklearn.load_model(best_model) error hai download krne main.
    # BAAD MEIN — dono functions mein ye karo:
    loan_prediction_model = load_best_model()
    prediction=loan_prediction_model.predict(data)
    output = np.where(prediction==1,'Y','N')
    result = {"prediction":output}
    return result


def generate_predictions_batch(data_input):
    # data = pd.DataFrame(data_input)
    experiment_name = config.EXPERIMENT_NAME
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    runs_df=mlflow.search_runs(experiment_ids=experiment_id,order_by=['metrics.f1_score DESC'])
    best_run=runs_df.iloc[0]
    best_run_id=best_run['run_id']
    best_model = f'runs:/{best_run_id}/{config.MODEL_NAME}'
    # loan_prediction_model=mlflow.sklearn.load_model(best_model)
    # BAAD MEIN — dono functions mein ye karo:
    loan_prediction_model = load_best_model()
    prediction=loan_prediction_model.predict(data_input)
    output = np.where(prediction==1,'Y','N')
    result = {"prediction":output}
    return result


    


if __name__=='__main__':
    generate_predictions()


# FastAPI se input aaya
#         ↓
# DataFrame banao
#         ↓
# MLflow mein experiment dhundo
#         ↓
# Saare runs fetch karo — f1 DESC sort
#         ↓
# Best run ka ID lo — iloc[0]
#         ↓
# Model URI banao — "runs:/ID/name"
#         ↓
# MLflow se poori pipeline load karo
#         ↓
# pipeline.predict(data)
#         ↓
# 1/0 → Y/N convert karo
#         ↓
# {"prediction": ["Y"]} return karo    