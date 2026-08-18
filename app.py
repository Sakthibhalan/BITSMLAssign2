import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, matthews_corrcoef, confusion_matrix, 
                             classification_report, roc_auc_score)

# --- Page Configuration ---
st.set_page_config(page_title="Breast Cancer Diagnostics", layout="centered")

st.title("Assignment 2: ML Classifier Dashboard")
st.write("Evaluate multiple classification models on Breast Cancer diagnostic data.")

# --- File Upload & Automatic Data Loading ---
# Widget is available but optional
user_csv = st.file_uploader("Upload your test_data.csv file (Optional)", type="csv")

# Logic: Use uploaded file if present, otherwise default to the local file
if user_csv is not None:
    user_csv.seek(0)  # Force reset the file buffer pointer
    eval_df = pd.read_csv(user_csv)
    st.success(f"✅ Successfully loaded custom file: {user_csv.name}")
elif os.path.exists("test_data.csv"):
    eval_df = pd.read_csv("test_data.csv")
    st.info("ℹ️ Automatically loaded default 'test_data.csv' from the repository.")
else:
    # Only halts if absolutely neither is available
    st.error("No dataset found! Please upload your test data.")
    st.stop()

# --- Model Dictionary ---
saved_models_dict = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "K-Nearest Neighbors": "model/knearest_neighbors.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}

# --- Dataset Preview ---
st.subheader(f"Dataset Preview (Evaluating {eval_df.shape[0]} rows)")
st.dataframe(eval_df.head(3))

# --- Feature & Target Extraction ---
target_name = 'diagnosis' 

if target_name in eval_df.columns:
    features = eval_df.drop(columns=[target_name])
    labels = eval_df[target_name]
    
    st.subheader("Model Selection")
    chosen_classifier = st.selectbox("Pick a trained model to evaluate:", list(saved_models_dict.keys()))
    
    try:
        # --- REAL PREDICTION CODE ---
        clf = joblib.load(saved_models_dict[chosen_classifier])
        predictions = clf.predict(features)
        
        # Get probability predictions (Required for AUC score)
        probabilities = clf.predict_proba(features)[:, 1]
        
        st.subheader(f"Results for {chosen_classifier}")
        
        # --- Metric Calculations ---
        acc_val = accuracy_score(labels, predictions)
        prec_val = precision_score(labels, predictions, zero_division=0)
        rec_val = recall_score(labels, predictions, zero_division=0)
        f1_val = f1_score(labels, predictions, zero_division=0)
        mcc_val = matthews_corrcoef(labels, predictions)
        
        # Error handling just in case the user uploads a test file with only one class
        try:
            auc_val = roc_auc_score(labels, probabilities)
        except ValueError:
            auc_val = 0.0 
        
        # --- Custom Metric Layout ---
        row1_1, row1_2, row1_3 = st.columns(3)
        row1_1.metric("Accuracy", f"{acc_val:.3f}")
        row1_2.metric("AUC Score", f"{auc_val:.3f}")
        row1_3.metric("MCC Score", f"{mcc_val:.3f}")
        
        row2_1, row2_2, row2_3 = st.columns(3)
        row2_1.metric("Precision", f"{prec_val:.3f}")
        row2_2.metric("Recall", f"{rec_val:.3f}")
        row2_3.metric("F1 Score", f"{f1_val:.3f}")
        
        st.divider()

        # --- Visualizations ---
        st.subheader("Confusion Matrix")
        matrix = confusion_matrix(labels, predictions)
        
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Purples', ax=ax)
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('Actual Class')
        st.pyplot(fig)
        
        st.subheader("Classification Report")
        report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose().style.format("{:.3f}"))

    except FileNotFoundError:
        st.warning(f"Could not find the model file at `{saved_models_dict[chosen_classifier]}`. Make sure you train your models and save them first!")
        
else:
    st.error(f"Error: Could not find the '{target_name}' column in the dataset. Ensure your CSV is formatted correctly.")