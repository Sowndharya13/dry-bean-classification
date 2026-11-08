import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

st.set_page_config(page_title="Dry Bean Classifier", layout="wide")
st.title("Dry Bean Classification Dashboard")
st.caption("Upload test data, pick a saved model, and view its performance.")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_resource
def load_encoder():
    return joblib.load("model/label_encoder.joblib")

st.sidebar.header("Controls")
model_name = st.sidebar.selectbox("Select model", list(MODEL_FILES.keys()))
uploaded = st.sidebar.file_uploader("Upload test CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV with all 16 feature columns and a 'Class' column.")
    st.stop()

df = pd.read_csv(uploaded)
if "Class" not in df.columns:
    st.error("The CSV must contain the target column 'Class'.")
    st.stop()

with st.expander("Preview uploaded data"):
    st.write(df.shape)
    st.dataframe(df.head())

le = load_encoder()
model = load_model(MODEL_FILES[model_name])

X = df.drop(columns=["Class"])
y_true = le.transform(df["Class"])

try:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
except Exception as e:
    st.error(f"Could not run the model on this data: {e}")
    st.stop()

try:
    auc = roc_auc_score(y_true, y_proba, multi_class="ovr")
except ValueError:
    auc = float("nan")

st.subheader(f"Evaluation metrics: {model_name}")
c = st.columns(6)
c[0].metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
c[1].metric("AUC", f"{auc:.4f}")
c[2].metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.4f}")
c[3].metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.4f}")
c[4].metric("F1", f"{f1_score(y_true, y_pred, average='weighted'):.4f}")
c[5].metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(le.classes_)))
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    report = classification_report(
        y_true, y_pred, labels=range(len(le.classes_)),
        target_names=le.classes_, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3))