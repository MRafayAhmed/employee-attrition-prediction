# employee-attrition-prediction
Predict employee attrition using machine learning &amp; HR analytics.
# Employee Attrition Prediction – Machine Learning Project

The project demonstrates:
- Structured problem-solving
- Clean ML pipeline design
- Preprocessing & feature engineering
- Model training & evaluation
- Explainability using SHAP
- Professional project structure and reproducibility

---

## 🚀 Project Objective

Employee attrition is a major challenge for HR teams.  
This project builds a **classification model** that predicts whether an employee is likely to leave the company.

The final result includes:
- A trained classification model  
- Explainable SHAP summary  
- Evaluation metrics  
- Preprocessing pipeline  
- End-to-end reproducible code  

---

## Dataset

IBM HR Analytics Employee Attrition Dataset  
Download from Kaggle:  
🔗 https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

## Create virtual environment:

bash
Copy code
python -m venv .venv
source .venv/Scripts/activate  # Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt

## How to Run the Project
1. Add dataset
Put the dataset file inside:

bash
Copy code
data/raw/
2. Run training
bash
Copy code
python src/train.py
This will:

## Preprocess data

Train the model
Evaluate performance
Generate SHAP explainability plot
Save outputs into artifacts/

## Outputs Generated
After running train.py, you will get:

File	Description
model.pkl	Final trained model
preprocessor.pkl	Scaler + OneHotEncoder pipeline
metrics.json	Accuracy, Precision, Recall, F1, AUC
shap_summary.png	Feature importance visualization

## Explainability Using SHAP
The project uses SHAP values to understand:
Which features have the strongest effect on attrition
Whether features increase or decrease likelihood of leaving
This improves model transparency for HR decision-makers.

## Tech Stack
Python 3.11+
Pandas, NumPy
Scikit-learn
imbalanced-learn (SMOTE)
SHAP
Matplotlib/Seaborn
Jupyter Notebook
