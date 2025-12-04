# src/predict.py
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/rf_attrition_model.joblib"

def predict_single(df_row: pd.DataFrame):
    bundle = joblib.load(MODEL_PATH)
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]

    X_trans = preprocessor.transform(df_row)
    proba = model.predict_proba(X_trans)[:,1]
    pred = model.predict(X_trans)
    return pred, proba

if __name__ == "__main__":
    example = pd.read_csv("data/example_employee.csv")  # create one row CSV
    pred, proba = predict_single(example)
    print("Pred:", pred, "Proba:", proba)
