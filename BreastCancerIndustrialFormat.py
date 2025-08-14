#####################################################################
# Required Packages
#####################################################################
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import(
    accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#####################################################################
# File Paths
#####################################################################

OUTPUT_PATH = "breast-cancer-wisconsin.csv"
MODEL_PATH = "bc_rf_pipeline.joblib"

#####################################################################
# Headers
#####################################################################

HEADERS = ["CodeNumber","ClumpThickness","UniformityCellSize","MarginalAdhesion",
           "SingleEpithelialCellSize","BareNuclei","BlandChromatin","NormalNucleoli",
           "Mitoses","CancerType"]

#####################################################################
# Function name :      read_data
# Description :        Read the data into pandas dtaframe
# Input :              path of CSV file
# OutPut :             Gives The Data
# Author :             Vipul Swapnil Barmukh
# Date :               12/08/2025   
######################################################################

def read_data(datapath):
    # Read The Data into pandas dataframe
    data = pd.read_csv(datapath, header=None)
    return data

#####################################################################
# Function name :      Handelling_missing_values
# Description :        Filter missing Values form the dataset
# Input :              Dataset with missing values
# OutPut :             dataset by removing missing values
# Author :             Vipul Swapnil Barmukh
# Date :               12/08/2025   
######################################################################

def handeling_missing_values(data,feature_headers):

    """
    Convert '?' to Nan and let SimpleImputer handle them inside the pipeline
    Keep only numeric columns in frame
    """
    # Replace '?' in the whole dataframe
    data = data.replace('?', np.nan)

    # Cast features to numeric
    data[feature_headers] = data[feature_headers].apply(pd.to_numeric, errors = 'coerce')

    return data

#####################################################################
# Function name :      split_dataset
# Description :        Split the dataset with train_percentage
# Input :              Dataset with related information
# OutPut :             Dataset after spliting
# Author :             Vipul Swapnil Barmukh
# Date :               12/08/2025   
######################################################################

def split_dataset(df, train_percentage, feature_headers,target_headers):
    """ Split dataset into train /test"""

    X_train,X_test,Y_train,Y_test = train_test_split(df[feature_headers], df[target_headers],train_size=train_percentage,random_state=42)

    return X_train, X_test, Y_train, Y_test


#####################################################################
# Function name :      dataset_statics
# Description :        Display Statics
# Author :             Vipul Swapnil Barmukh
# Date :               12/08/2025   
######################################################################

def dataset_statics(dataset):
    """ Print basic stats"""

    print(dataset.describe())

#####################################################################
# Function name :      dataset_pipeline
# Description :        Build a pipeline
#SimpleImputer :       replace missing with median
# Author :             Vipul Swapnil Barmukh
# Date :               12/08/2025   
######################################################################

def build_pipeline():

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier
         (
            n_estimators= 300,
            random_state=42,
            n_jobs=-1,
            class_weight = None
        ))
    ])

    return pipe

#####################################################################
# Function name :      train_pipeline
# Description :        train the pipeline
# Author :             Vipul Swapnil Barmukh
# Date :               12/08/2025   
######################################################################

def train_pipeline(pipeline, X_train, Y_train):
    pipeline.fit(X_train,Y_train)
    return pipeline

#####################################################################
# Function name :      save_model
# Description :        Save The model
# Author :             Vipul Swapnil Barmukh
# Date :               12/08/2025   
######################################################################

def save_model(model, path= MODEL_PATH):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

#####################################################################
# Function name :      load_model
# Description :        Load the train model
# Author :             Vipul Swapnil Barmukh
# Date :               12/08/2025   
######################################################################

def load_model(path = MODEL_PATH):
    model = joblib.load(path)
    print(f"Model load from {path}")
    return model

#####################################################################
# Function name :      plot_feature_importances
# Description :        Display The features importance
# Author :             Vipul Swapnil Barmukh
# Date :               12/08/2025   
######################################################################
def plot_features_importances(model, feature_names, title ="Feature Importances(Random Forest)"):
    if hasattr(model, "named_steps") and "rf" in model.named_steps:
        rf = model.named_steps["rf"]
        importances = rf.feature_importances_
    elif hasattr(model, "features_importances"):
        importances = model.feature_importances_
    else:
        print("Feature importances not available for this model")
        return
    
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(8,4))
    plt.bar(range(len(importances)), importances[idx])
    plt.xticks(range(len(importances)),[feature_names[i] for i in idx],rotation = 45, ha='right')
    plt.ylabel("Importances")
    plt.title(title)
    plt.tight_layout()
    plt.show()

#####################################################################
# Function name :      main
# Description :        Main function from where execution starts
# Author :             Vipul Swapnil Barmukh
# Date :               12/08/2025   
######################################################################

def main():
    print("hello")
    #Load CSV
    dataset = pd.read_csv(OUTPUT_PATH)

    # 2) Drop unnecessary columns 
    dataset.drop(columns=HEADERS[0], inplace=True)

    # 3) Basic stats
    dataset_statics(dataset)

    # 4) Prepare features/target
    feature_headers = HEADERS[1:-1]     # Drop CodeNumber, keep all features
    target_headers = HEADERS[-1]        # CancerType (benign = 2, maligant = 4)

    # 5) Handle '?' and coerce to numeric; imputation will happen inside Pipeline
    dataset = handeling_missing_values(dataset, feature_headers)

    # 6) Split
    X_train, X_test, Y_train, Y_test = split_dataset(dataset,0.7,feature_headers,target_headers)

    print(f"X_train shape : {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"Y_train shape : {Y_train.shape}")
    print(f"Y_test shape : {Y_test.shape}")

    # 7) Build + Train Pipeline
    pipeline = build_pipeline()
    trained_model = train_pipeline(pipeline, X_train, Y_train)


    # 8)Predictions
    predictions = trained_model.predict(X_test)

    # 9) Metrics
    print(f"Training Accuracy : {accuracy_score(Y_train, trained_model.predict(X_train))}")
    print(f"Testing Accuracy : {accuracy_score(Y_test, predictions)}")
    print(f"Classification Report :\n{classification_report(Y_test, predictions)}")
    print(f"Confusion Matrix :\n {confusion_matrix(Y_test, predictions)}")

    # 10) Feature importances(tree-based)
    plot_features_importances(trained_model, feature_headers, title="Feature Importances (RF)")

    # 11) Save model (Pipeline) using joblib
    save_model(trained_model, MODEL_PATH)

    # 12) Load model and test a sample
    loaded_model = load_model(MODEL_PATH)
    sample = X_test.iloc[[0]]
    pre_loaded = loaded_model.predict(sample)
    print(f"Loaded model prediction for sample : {pre_loaded[0]}")


#############################################################################################################
# Application Starter
#############################################################################################################
if __name__== "__main__":
    main()







   


