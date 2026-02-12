# Task 15 — End-to-End Machine Learning Pipeline

## Objective
Build a complete machine learning pipeline using scikit-learn that includes preprocessing, model training, evaluation, and model saving.

---

## Tools Used
- Python
- Pandas
- NumPy
- Scikit-learn (Pipeline, ColumnTransformer)
- Joblib
- Jupyter Notebook / Google Colab

---

## Dataset
Breast Cancer Dataset from scikit-learn.

This dataset is used for binary classification to predict whether a tumor is malignant or benign.

---

## Steps Performed

### 1. Data Loading
Loaded the Breast Cancer dataset using `sklearn.datasets`.

### 2. Feature Identification
Separated numerical and categorical features.
All features in this dataset are numerical.

### 3. Preprocessing
Used `ColumnTransformer` to apply:
- StandardScaler to numerical features

### 4. Pipeline Creation
Created a complete ML pipeline using:
- ColumnTransformer (preprocessing)
- Logistic Regression (model)

### 5. Train-Test Split
Split dataset into training and testing sets using `train_test_split`.

### 6. Model Training
Trained the pipeline using the training data.

### 7. Prediction
Generated predictions on the test dataset.

### 8. Evaluation Metrics
Model performance evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score

### 9. Model Saving
Saved the trained pipeline using `joblib`:
`ml_pipeline.pkl`

---

## Key Learning Outcomes
- Understanding ML pipelines
- Using ColumnTransformer for preprocessing
- Preventing data leakage using pipelines
- Building production-style ML workflows
- Saving trained models for deployment

---

## Repository Structure

```
AI-ML-Internship-Task-15/
 ├── Task_15_ML_Pipeline.ipynb
 ├── ml_pipeline.pkl
 └── README.md
```

---

## Conclusion
This task demonstrates how to build an end-to-end machine learning pipeline combining preprocessing and model training into a single workflow, similar to real-world ML systems.

---
