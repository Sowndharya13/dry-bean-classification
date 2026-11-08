# Dry Bean Classification

Five classification models trained on the UCI Dry Bean dataset and served through an interactive Streamlit dashboard.

**Live app:** https://dry-bean-classification-8pe8udgjsg8qcw3wmfrcrp.streamlit.app/

## a. Problem statement

Dry beans of different varieties look similar to the eye, and sorting them by hand is slow and inconsistent. The goal of this project is to classify a bean into one of seven varieties using only measurements of its shape and size taken from images. Five classifiers (Logistic Regression, Decision Tree, KNN, Naive Bayes and SVM) are trained on the same data and compared with six metrics (Accuracy, AUC, Precision, Recall, F1 and MCC) to find the best model.

## b. Dataset description

- **Source:** UCI Machine Learning Repository, [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **Instances:** 13,611 beans
- **Features:** 16 numeric features (12 or more required)
- **Target:** `Class`, with 7 categories: Seker, Barbunya, Bombay, Cali, Dermason, Horoz and Sira (multi-class classification)
- **Missing values:** none

**Features:** Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4. These describe the size and shape of each bean (dimensions, roundness, compactness and similar measures).

**Class distribution (full dataset):** Dermason 3,546, Sira 2,636, Seker 2,027, Horoz 1,928, Cali 1,630, Barbunya 1,322, Bombay 522. The classes are imbalanced, with Dermason about seven times more frequent than Bombay.

**Preprocessing and split:**
- The target was label-encoded (the encoder is saved as `model/label_encoder.joblib`).
- The data was split 80/20 into train and test sets with stratification (`random_state=42`), so every class keeps its proportion in both sets.
- The test set (2,723 rows) is saved as `test_data.csv` and is what the Streamlit app uses.
- Logistic Regression, KNN and SVM use a `StandardScaler` inside a scikit-learn `Pipeline`, because they are sensitive to feature scale. Decision Tree and Naive Bayes use the raw features.
- Because the classes are imbalanced, Precision, Recall and F1 are weighted averages, and AUC is computed one-vs-rest (OvR).

## c. GitHub repository

https://github.com/Sowndharya13/dry-bean-classification

```
dry-bean-classification/
|-- app.py                       Streamlit application
|-- requirements.txt             Dependencies for deployment
|-- README.md
|-- test_data.csv                Held-out test set (with target column)
|-- Dry_Bean_Asignment.ipynb     Training and evaluation notebook
|-- Dry_Bean_Dataset.xlsx        Original dataset
`-- model/                       Saved trained models
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- support_vector_machine.joblib
    `-- label_encoder.joblib
```

## d. Models used

All five models were trained on the same training split and evaluated on the same test set.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9214 | 0.9948 | 0.9222 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree | 0.8917 | 0.9448 | 0.8912 | 0.8917 | 0.8913 | 0.8691 |
| KNN | 0.9166 | 0.9833 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Naive Bayes | 0.7639 | 0.9672 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| Support Vector Machine | 0.9221 | 0.9949 | 0.9224 | 0.9221 | 0.9222 | 0.9059 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performed very strongly (accuracy 0.9214, AUC 0.9948, MCC 0.9050) and came within a hair of the best model, even though it is the simplest. This suggests the bean classes are close to linearly separable once the features are standardized. It is also fast to train and easy to interpret. |
| Decision Tree | Weaker than the other models apart from Naive Bayes (accuracy 0.8917) and has the lowest AUC among the four stronger models (0.9448). A single tree tends to overfit and gives coarse, almost hard class probabilities, which lowers the ranking-based AUC. It does not need scaling, but it is clearly less accurate than the smoother models. |
| KNN | Good results (accuracy 0.9166, AUC 0.9833, MCC 0.8992), slightly behind Logistic Regression and SVM. Scaling is essential here because the features have very different ranges. Its predictions depend on local neighbourhoods, so it struggles a little where similar varieties overlap. It is also the slowest to predict because it compares against stored training points. |
| Naive Bayes | The weakest model by a clear margin (accuracy 0.7639, MCC 0.7154). Many features, such as Area, Perimeter, ConvexArea and the axis lengths, are strongly correlated, which breaks the independence assumption of Naive Bayes. Its AUC is still reasonable (0.9672), so it ranks classes well but turns those probabilities into wrong final labels more often. |
| Support Vector Machine | The best model on every metric (accuracy 0.9221, AUC 0.9949, F1 0.9222, MCC 0.9059). The RBF kernel handles the non-linear boundaries between similar bean types, and standardization keeps all features on the same scale. The lead over Logistic Regression is small, under 0.001 on accuracy. |

### Overall winner

**Support Vector Machine** is the winner for this dataset. It has the highest Accuracy, AUC, Precision, Recall, F1 and MCC. Its margin over Logistic Regression is tiny, so Logistic Regression is a very good simpler alternative when training and prediction speed matter. Naive Bayes is the clear underperformer because of the correlated features.

## Streamlit app

The deployed app lets you:
1. **Upload a CSV** of test data (use `test_data.csv`). It must contain all 16 feature columns and the `Class` column.
2. **Select a model** from a dropdown of the five saved models.
3. **View evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC.
4. **View a confusion matrix and a classification report** for the selected model.

The app loads the saved models from `model/` and only runs predictions. It never retrains on the uploaded data.

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, joblib and Streamlit. The models were trained with scikit-learn 1.9.1, which is pinned in `requirements.txt`.
