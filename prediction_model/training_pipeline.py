import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import xgboost as xgb
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
import prediction_model.processing.preprocessing as pp 
import prediction_model.pipeline as pipe
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import dagshub
import mlflow
import os
#mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_tracking_uri(config.TRACKING_URI)
dagshub.init(repo_owner='rakeshcpr011', repo_name='MLOps-E2E-POC-i-mubahsir-hasan', mlflow=True)


def get_data(input):
    data=load_dataset(input)
    x=data[config.FEATURES]
    y=data[config.TARGET].map({'N':0,'Y':1})
    return x,y
   

X,Y=get_data(config.TRAIN_FILE)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
   

# Define the search space
search_space = {
    'max_depth': hp.choice('max_depth', np.arange(3, 10, dtype=int)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'n_estimators': hp.choice('n_estimators', np.arange(50, 300, 50, dtype=int)),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    'gamma': hp.uniform('gamma', 0, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1)
}


def objective(params):
    # Create an XGBoost classifier with the given hyperparameters
    
    clf = xgb.XGBClassifier(
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        gamma=params['gamma'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    # Define the complete pipeline with preprocessing and model
    classification_pipeline = Pipeline(
        [
            ('DomainProcessing', pp.DomainProcessing(variable_to_modify=config.FEATURE_TO_MODIFY, variable_to_add=config.FEATURE_TO_ADD)),
            ('MeanImputation', pp.MeanImputer(variables=config.NUM_FEATURES)),
            ('ModeImputation', pp.ModeImputer(variables=config.CAT_FEATURES)),
            ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
            ('LabelEncoder', pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
            ('LogTransform', pp.LogTransforms(variables=config.LOG_FEATURES)),
            ('MinMaxScale', MinMaxScaler()),
            ('XGBoostClassifier', clf)
        ]
    )
    
   
    # Fit the pipeline
    # mlflow.xgboost.autolog()
    mlflow.set_experiment("loan_prediction_model")

    with mlflow.start_run(nested=True):
        # Fit the pipeline
        classification_pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = classification_pipeline.predict(X_test)
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        
        # Log metrics manually
        mlflow.log_metrics({
            'f1_score': f1,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision
        })

        mlflow.sklearn.log_model(classification_pipeline,config.MODEL_NAME)
    return {'loss': 1-f1, 'status': STATUS_OK}
    


trials = Trials()

best_params = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=5, trials=trials)


print("Best hyperparameters:", best_params)

# =================================
# only for local caching of best model — predict.py mein bhi ye function banao:
# ✅ YE ADD KARO — Best model locally save karo
best_trial = min(trials.results, key=lambda x: x['loss'])
print(f"Best F1: {1 - best_trial['loss']}")


# Best params se final model banao aur save karo
best_clf = xgb.XGBClassifier(
    max_depth=int(best_params['max_depth']),
    learning_rate=best_params['learning_rate'],
    n_estimators=int(best_params['n_estimators']),
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    gamma=best_params['gamma'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda'],
    eval_metric='mlogloss'
)

final_pipeline = Pipeline([
    ('DomainProcessing', pp.DomainProcessing(
        variable_to_modify=config.FEATURE_TO_MODIFY,
        variable_to_add=config.FEATURE_TO_ADD)),
    ('MeanImputation', pp.MeanImputer(variables=config.NUM_FEATURES)),
    ('ModeImputation', pp.ModeImputer(variables=config.CAT_FEATURES)),
    ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
    ('LabelEncoder', pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
    ('LogTransform', pp.LogTransforms(variables=config.LOG_FEATURES)),
    ('MinMaxScale', MinMaxScaler()),
    ('XGBoostClassifier', best_clf)
])

final_pipeline.fit(X_train, y_train)

# Local mein save karo
import joblib
os.makedirs(os.path.join(config.PACKAGE_ROOT, 'trained_models'), exist_ok=True)
model_path = os.path.join(config.PACKAGE_ROOT, 'trained_models', 'cached_model.pkl')
joblib.dump(final_pipeline, model_path)
print(f"✅ Model saved locally at: {model_path}")


#=================
# fmin() → 5 baar objective() call karta hai
#               ↓
#          Har trial mein:
#          naya XGBoost → Pipeline → fit → f1 calculate
#               ↓
#          MLflow mein log: params + metrics + model
#               ↓
#          loss = 1-f1 → hyperopt ko do
#               ↓
#          Hyperopt next better params choose karta hai
#               ↓
# 5 trials complete → best_params print
#               ↓
# predict.py → MLflow se best f1 wala model load



