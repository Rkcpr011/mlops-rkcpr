from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from prediction_model.predict import generate_predictions, generate_predictions_batch
from prediction_model.config import config
import mlflow
import io
import os
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator

mlflow.set_tracking_uri(config.TRACKING_URI)

app = FastAPI(
    title="Loan Prediction App using FastAPI - MLOps",
    description="MLOps Demo",
    version='1.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

Instrumentator().instrument(app).expose(app)


class LoanPrediction(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str


@app.get("/")
def index():
    return {"message": "Welcome to the MLOps Loan Prediction app"}


@app.post("/prediction_api")
def predict(loan_details: LoanPrediction):
    data = loan_details.model_dump()
    prediction = generate_predictions([data])["prediction"][0]
    pred = "Approved" if prediction == "Y" else "Rejected"
    return {"status": pred}


@app.post("/prediction_ui")
def predict_gui(
    Gender: str,
    Married: str,
    Dependents: str,
    Education: str,
    Self_Employed: str,
    ApplicantIncome: float,
    CoapplicantIncome: float,
    LoanAmount: float,
    Loan_Amount_Term: float,
    Credit_History: float,
    Property_Area: str
):
    cols = config.FEATURES
    input_data = [Gender, Married, Dependents, Education, Self_Employed,
                  ApplicantIncome, CoapplicantIncome, LoanAmount,
                  Loan_Amount_Term, Credit_History, Property_Area]
    data_dict = dict(zip(cols, input_data))
    prediction = generate_predictions([data_dict])["prediction"][0]
    pred = "Approved" if prediction == "Y" else "Rejected"
    return {"status": pred}


@app.post("/batch_prediction")
async def batch_predict(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content), index_col=False)

    if not all(col in df.columns for col in config.FEATURES):
        return {"error": "CSV file does not contain the required columns."}

    predictions = generate_predictions_batch(df)["prediction"]
    df['Prediction'] = predictions
    result = df.to_csv(index=False)

    # Local folder mein save karo
    os.makedirs("batch_outputs", exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"batch_outputs/{file.filename}_{current_datetime}.csv"
    with open(output_path, 'w') as f:
        f.write(result)

    return StreamingResponse(
        io.BytesIO(result.encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"}
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)