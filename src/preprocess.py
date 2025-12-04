# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

def build_preprocessor(df: pd.DataFrame, numeric_features=None, categorical_features=None):
    if numeric_features is None:
        numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

    # remove target if included
    if "Attrition" in numeric_features:
        numeric_features.remove("Attrition")
    if "Attrition" in categorical_features:
        categorical_features.remove("Attrition")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),

        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_features, categorical_features

def fit_preprocessor(df, save_path=None):
    preprocessor, num_feats, cat_feats = build_preprocessor(df)
    preprocessor.fit(df.drop(columns=["Attrition"]))
    if save_path:
        joblib.dump({"preprocessor": preprocessor, "num_feats": num_feats, "cat_feats": cat_feats}, save_path)
    return preprocessor, num_feats, cat_feats
def get_feature_names(preprocessor):
    ohe = preprocessor.named_transformers_['cat']['onehot']
    cat_feature_names = ohe.get_feature_names_out()
    num_feature_names = preprocessor.named_transformers_['num'].feature_names_in_
    return list(num_feature_names) + list(cat_feature_names)

