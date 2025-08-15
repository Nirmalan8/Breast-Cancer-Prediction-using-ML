# Breast Cancer Prediction using Machine Learning Algorithms

##  Project Overview
This project predicts whether a breast tumor is **benign** or **malignant** using various **machine learning algorithms**. The dataset includes multiple features related to breast cancer diagnosis. The objective is to build, train, and evaluate different models for accurate predictions.

---

## Dataset
- **Source:** Breast Cancer Wisconsin (Diagnostic) Dataset (from `sklearn.datasets`)
- **Features:** Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Symmetry, etc.
- **Target:**
  - `0` → Malignant
  - `1` → Benign

---

## Technologies and Libraries
- Python
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `xgboost`

---

##  Steps Performed
1. **Data Loading & Exploration**
   - Load dataset using `sklearn.datasets`
   - Analyze dataset structure and summary statistics

2. **Data Preprocessing**
   - Check missing values
   - Normalize / standardize features
   - Split into training and testing sets

3. **Model Building**
   - Implemented:
     - Logistic Regression
     - K-Nearest Neighbors (KNN)
     - Support Vector Machine (SVM)
     - Random Forest
     - XGBoost

4. **Model Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-Score
   - Confusion matrix visualization

5. **Model Comparison**
   - Compared accuracy scores of all models
   - Selected best-performing model

---

##  Visualizations
- Feature correlation heatmap
- Confusion matrix for each model
- Accuracy comparison bar chart

---

##  Results
- **Ensemble models** like Random Forest and XGBoost achieved the highest accuracy.
- Logistic Regression also performed well.

---

