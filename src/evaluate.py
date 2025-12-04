# src/evaluate.py
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from utils import save_fig

MODEL_PATH = "models/rf_attrition_model.joblib"
DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
REPORT_DIR = "reports/figures"

def evaluate():
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    preprocessor = model_bundle["preprocessor"]

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"].apply(lambda v: 1 if v == "Yes" else 0)
    X_trans = preprocessor.transform(X)

    proba = model.predict_proba(X_trans)[:, 1]
    preds = model.predict(X_trans)
    print(classification_report(y, preds))
    auc = roc_auc_score(y, proba)
    print("AUC:", auc)

    # ROC
    fpr, tpr, _ = roc_curve(y, proba)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr)
    ax.plot([0,1], [0,1], '--')
    ax.set_title("ROC Curve")
    save_fig(fig, f"{REPORT_DIR}/roc_curve_eval.png")
    plt.close(fig)

if __name__ == "__main__":
    evaluate()
