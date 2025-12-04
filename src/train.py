# src/train.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt

from preprocess import fit_preprocessor, build_preprocessor
from utils import load_data, train_test_split_xy, save_fig

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
MODEL_DIR = "models"
REPORT_DIR = "reports/figures"

def train_and_save(random_state=42):
    df = load_data(DATA_PATH)
    # basic cleanup: drop EmployeeCount, Over18, StandardHours if present (non-informative)
    for c in ["EmployeeCount", "Over18", "StandardHours"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Build preprocessor and transform
    preprocessor, num_feats, cat_feats = fit_preprocessor(df)
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"].apply(lambda v: 1 if v == "Yes" else 0)
    X_trans = preprocessor.transform(X)

    # Train/test split manually to align with utils
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_trans, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Handle imbalance with SMOTE (on training data)
    sm = SMOTE(random_state=random_state)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    # RandomForest
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    rf.fit(X_train_sm, y_train_sm)

    # Evaluate
    preds = rf.predict(X_test)
    proba = rf.predict_proba(X_test)[:, 1]
    print("Classification report:")
    print(classification_report(y_test, preds))
    auc = roc_auc_score(y_test, proba)
    print("ROC AUC:", auc)

    # Save model + preprocessor
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump({"model": rf, "preprocessor": preprocessor, "num_feats": num_feats, "cat_feats": cat_feats}, os.path.join(MODEL_DIR, "rf_attrition_model.joblib"))
    print("Saved model.")

    # Feature importance (approx): need feature names after one-hot encoding
    # derive feature names
    num_cols = num_feats
    # build onehot names
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_cols = cat_feats
    ohe_names = list(ohe.get_feature_names_out(cat_cols))
    feature_names = list(num_cols) + ohe_names

    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:30]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in sorted_idx[::-1]], importances[sorted_idx[::-1]])
    ax.set_title("Top feature importances")
    save_fig(fig, os.path.join(REPORT_DIR, "feature_importance.png"))
    plt.close(fig)

    # SHAP summary (TreeExplainer)
    # For binary classification with tree models, shap_values returns a list for each class
    # Use a sample of data for SHAP to avoid memory issues
    sample_size = min(100, X_train.shape[0])
    X_sample = X_train[:sample_size]
    
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)
    
    # For binary classification, take class 1 (attrition)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values
    
    shap.summary_plot(shap_vals, X_sample, feature_names=feature_names, show=False)
    fig = plt.gcf()
    save_fig(fig, os.path.join(REPORT_DIR, "shap_summary.png"))
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(fpr, tpr)
    ax2.plot([0, 1], [0, 1], linestyle="--")
    ax2.set_title("ROC curve - RandomForest")
    ax2.set_xlabel("FPR")
    ax2.set_ylabel("TPR")
    save_fig(fig2, os.path.join(REPORT_DIR, "roc_curve.png"))
    plt.close(fig2)

    # Save a quick text report
    report_path = os.path.join("reports", "model_report.txt")
    with open(report_path, "w") as f:
        f.write("ROC AUC: {:.4f}\n".format(auc))
        f.write("Classification report:\n")
        f.write(classification_report(y_test, preds))
    print("Reports saved.")

if __name__ == "__main__":
    train_and_save()
