# src/utils.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def train_test_split_xy(df, target_col='Attrition', test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col].apply(lambda v: 1 if v == 'Yes' else 0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def save_fig(fig, path):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    fig.savefig(path, bbox_inches='tight', dpi=150)
