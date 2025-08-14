# Breast-Cancer-Prediction_CaseStudy
A machine learning project that predicts whether a breast tumor is benign or malignant using the Wisconsin Breast Cancer dataset.

## 📋 Overview

This project implements a breast cancer classification system using Random Forest algorithm with scikit-learn. The system processes medical data features and predicts cancer type with high accuracy.

## 🔧 Dependencies

Make sure you have the following Python packages installed:

bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib


*Required Libraries:*
- pandas - Data manipulation and analysis
- numpy - Numerical computing
- matplotlib - Data visualization
- seaborn - Statistical data visualization
- scikit-learn - Machine learning algorithms
- joblib - Model serialization

## 📊 Dataset Information

*Dataset:* Wisconsin Breast Cancer Dataset (breast-cancer-wisconsin.csv)

*Features (9 total):*
- ClumpThickness
- UniformityCellSize
- UniformityCellShape
- MarginalAdhesion
- SingleEpithelialCellSize
- BareNuclei
- BlandChromatin
- NormalNucleoli
- Mitoses

*Target Variable:* CancerType
- 2 = Benign (non-cancerous)
- 4 = Malignant (cancerous)

## 🔄 Workflow

### 1. Data Loading & Preprocessing
- Loads the CSV dataset
- Removes the CodeNumber column
- Handles missing values by replacing '?' with NaN
- Converts features to numeric format

### 2. Data Analysis
- Displays basic dataset statistics
- Shows data distribution and summary

### 3. Data Splitting
- Splits data into training (70%) and testing (30%) sets
- Uses stratified sampling for balanced classes

### 4. Model Training
- *Algorithm:* Random Forest Classifier
- *Configuration:*
  - 300 estimators (trees)
  - Median imputation for missing values
  - Parallel processing enabled
- *Pipeline:* Imputation → Random Forest Classification

### 5. Model Evaluation
- Training and testing accuracy scores
- Detailed classification report
- Confusion matrix analysis
- Feature importance visualization

### 6. Model Persistence
- Saves trained model using joblib
- Demonstrates model loading and prediction

## 🚀 Running the Project

### Prerequisites
1. Ensure you have the required dataset: breast-cancer-wisconsin.csv
2. Install all dependencies listed above

## Expected Output:
Training Accuracy : 1.0

Testing Accuracy : 0.97

Classification Report :

              precision    recall  f1-score   support

           2       0.98      0.97      0.97       135
           4       0.95      0.96      0.95        75

    accuracy                           0.97       210
    macro          0.96      0.97      0.96       210
    weighted avg   0.97      0.97      0.97       210

Confusion Matrix :

             [[131   4]
              [  3 72]]
 
Model saved to bc_model_pipeline.joblib

Model loaded from path : bc_model_pipeline.joblib

Loaded model prediction for sample : 4



## 👨‍💻 Author

*Vipul Barmukh*  
Date: 12/08/2025
