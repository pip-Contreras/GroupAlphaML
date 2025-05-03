'''
Author: Group AlphaML
April 17, 2025
Description: Runs Logistic Regression model on the datasets.
Dataset Sources:
1. 2015 Dataset: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
2. 2022 Dataset: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
'''

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

# Ensure results directory exists

os.makedirs("results/model_performance", exist_ok=True)

# Load Data

def get_data():
    """Loads two datasets into json divided into features and labels."""
    df_2015 = pd.read_csv("processed_data/df_2015.csv")
    X_2015 = df_2015.drop(columns=["HeartDiseaseorAttack"]).values #features
    y_2015 = df_2015["HeartDiseaseorAttack"].values #target

    df_2022 = pd.read_csv("processed_data/df_2022.csv")
    X_2022 = df_2022.drop(columns=["HadHeartAttack"]).values #features
    y_2022 = df_2022["HadHeartAttack"].values #target

    return [
        {"name": "df_2015", "X": X_2015, "y": y_2015},
        {"name": "df_2022", "X": X_2022, "y": y_2022}
    ]

# SMOTE to balance classes

def oversample_data(X, y):
    """Oversamples positive class using SMOTE.
    
    Parameters
    ----------
    X_train : array-like
        Training feature set.
    y_train : array-like
        Training target labels.
    """
    smote = SMOTE(random_state=1) #oversample strategy
    return smote.fit_resample(X, y)


# Logistic Regression classifier

def logistic_regression_classifier(X, y, dataset_name):
    """
    Train an estimator.

    Parameters
    ----------
    dataset_name : str
        Name of file where the results will be stored.
    x : array-like
        Training feature set.
    y : array-like
        Training target labels.
    """
    print(f"\n Training Logistic Regression on {dataset_name}...\n")
    start = time.time() #track training time

    #Cross-validation
    lr_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    #Base Classifier
    lr_model = LogisticRegression(solver="liblinear", max_iter=1000)

    #Grid Search
    lr_param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "class_weight": [None, "balanced"]
    }

    #GridSearchVC to tune hyperparameters
    lr_grid = GridSearchCV(
        estimator=lr_model,
        param_grid=lr_param_grid,
        cv=lr_skf,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    lr_grid.fit(X, y) #train model with cross-validation
    
    #Best model from grid
    lr_best_model = lr_grid.best_estimator_
    lr_y_pred = lr_best_model.predict(X) #predicted classes
    lr_y_proba = lr_best_model.predict_proba(X)[:, 1] #predicted probabilities

    end = time.time()
    runtime = end - start #total training time

# Classification Report

    #text classification
    print("Classification Report:")
    lr_report = classification_report(y, lr_y_pred, digits=4)
    print(lr_report)

    #save classification report to file
    with open(f"results/model_performance/lr_report_{dataset_name}.txt", "w") as f:
        f.write(f"Logistic Regression Report ({dataset_name})\n\n")
        f.write(lr_report)
        f.write(f"\nBest Params: {lr_grid.best_params_}\n")
        f.write(f"Runtime (s): {runtime:.2f}\n")

# Confusion Matrix
    
    lr_cm = confusion_matrix(y, lr_y_pred)
    print("Confusion Matrix:")
    print(lr_cm)

    #plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(lr_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"results/model_performance/lr_confusion_matrix_{dataset_name}.png", dpi=300)
    plt.show()

# ROC Curve

    lr_fpr, lr_tpr, _ = roc_curve(y, lr_y_proba)
    lr_roc_auc = auc(lr_fpr, lr_tpr)

    #plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(lr_fpr, lr_tpr, label=f"AUC = {lr_roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {dataset_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"results/model_performance/lr_roc_curve_{dataset_name}.png", dpi=300)
    plt.show()

    print(f"Best Parameters: {lr_grid.best_params_}")
    print(f"Runtime: {runtime:.2f} seconds")

    return lr_best_model #return model for optional reuse

# Run classifier for each dataset with oversampling
datasets = get_data()
for data in datasets:
    X_res, y_res = oversample_data(data["X"], data["y"])
    logistic_regression_classifier(X_res, y_res, data["name"])






