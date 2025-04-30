'''
Author: Group AlphaML
April 17, 2025
Description: Classifiers for heart attack datasets.
Dataset Sources:
1. 2015 Dataset: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
2. 2022 Dataset: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
'''
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import time
from imblearn.over_sampling import SMOTE


# =============================================================================
# Data
# =============================================================================
def get_data():
    df_2015 = pd.read_csv("processed_data/df_2015.csv")
    X_2015 = df_2015.drop(columns=["HeartDiseaseorAttack"]).values
    y_2015 = df_2015["HeartDiseaseorAttack"].values

    df_2022 = pd.read_csv("processed_data/df_2022.csv")
    X_2022 = df_2022.drop(columns=["HadHeartAttack"]).values
    y_2022 = df_2022["HadHeartAttack"].values

    datasets = [
        {
            "name": "2015",
            "X": X_2015,
            "y": y_2015
        },
        {
            "name": "2022",
            "X": X_2022,
            "y": y_2022
        }
    ]

    return datasets



# =============================================================================
# Helper Functions
# =============================================================================
def oversample_class(X_train, y_train):
    smote = SMOTE(random_state = 1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def save_class_ratios(y, file_name, title):
    print(file_name)
    print(title)
    # Get unique class labels and their counts
    classes, counts = np.unique(y, return_counts=True)

    # Calculate ratios
    ratios = counts / counts.sum()
        
    with open(file_name + ".txt", "a") as file:
        file.write(title + "\n")
        for cls, count, ratio in zip(classes, counts, ratios):
            file.write(f"Class {cls}: Count = {count}, Ratio = {ratio:.2%}\n")
        print("\n")

def plot_roc_curve(y_test, y_pred_proba, filename="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score: 2f})')
    plt.plot([0,1], [0,1], linestyle="--", color="gray") # ref line

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(filename, dpi=300)
    print(f"ROC curve saved as {filename}")



# =============================================================================
# Training Functions
# =============================================================================
def get_estimators():
    estimators = [
        {
            "model":{
                "name": "XGBoost",
                "estimator": XGBClassifier(),
            },
            "params": {
                "n_estimators": [50, 100, 200, 500],
                "learning_rate": [0.001, 0.01, 0.1, 0.5],
                "max_depth": [3, 5, 7]
            },
            "best_model": [],
            "metrics":{
                "accuracy": 0.0,
                "precision": 0.0,
                "f1": 0.0,
                "recall": 0.0,
                "confusion_matrix": [],
                "runtime": 0.0
            }
        },
        {
            "model":{
                "name": "XGBoost_Weighted",
                "estimator": XGBClassifier(scale_pos_weight=10),
            },
            "params": {
                "n_estimators": [50, 100, 200, 500],
                "learning_rate": [0.001, 0.01, 0.1, 0.5],
                "max_depth": [3, 5, 7]
            },
            "best_model": [],
            "metrics":{
                "accuracy": 0.0,
                "precision": 0.0,
                "f1": 0.0,
                "recall": 0.0,
                "confusion_matrix": [],
                "runtime": 0.0
            }
        },
    ]
    return estimators


def set_metrics(file_name, estimator, X_train, X_test, y_train, y_test):
    # Train estimator
    start_time = time.perf_counter()   
    estimator["model"]["estimator"].fit(X_train, y_train)
    end_time = time.perf_counter()
    y_pred = estimator["model"]["estimator"].predict(X_test)

    # calculate metrics
    estimator["metrics"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "runtime": end_time - start_time,
    }
    y_pred_proba = estimator["model"]["estimator"].predict_proba(X_test)[:, 1]
    model_name = estimator["model"]["name"].replace(" ", "_")
    plot_roc_curve(y_test, y_pred_proba, filename=f'{file_name}_roc_curve_{model_name}.png')


def evaluate_estimators(file_name, estimators, X_train, X_test, y_train, y_test):
    # Stratified k-fold cross-validation is good for unbalanced datasets
    n_splits = 5 # 5 is a good balance for medium size datasets.
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

    try:
        for estimator in estimators:
            grid_search = GridSearchCV(
                estimator["model"]["estimator"], # estimator
                estimator["params"],    # grid of hyperparameters
                cv=stratified_kfold, # cross-validation
                n_jobs=-1       # to use all cores
            )
            # train estimators
            grid_search.fit(X_train, y_train)
            # find best tunning
            estimator["best_model"] = grid_search.best_estimator_
            # evaluate best tunning
            set_metrics(file_name, estimator, X_train, X_test, y_train, y_test)
    except KeyboardInterrupt:
        exit()
    except Exception as e:
        print("Error optimizing model: ", e)


import numpy as np
def classify():
    for dataset in get_data():
        # splitting datasets
        X_train, X_test, y_train, y_test = train_test_split(dataset["X"], dataset["y"], test_size=0.95, random_state=1, stratify=dataset["y"])
        
        # oversampling
        X_train_resampled, y_train_resampled = oversample_class(X_train, y_train)  
        estimators = get_estimators()

        # Save data set data
        file_name = f"results/model_performance/{dataset['name']}"
        title = f"Training Dataset: Original dataset (80%)" 
        title_oversampled = f"Training Dataset: Oversamples dataset (80%)" 
        save_class_ratios(y_train_resampled, file_name, title)
        save_class_ratios(y_train, file_name, title_oversampled)

        # evaluate original dataset
        evaluate_estimators(file_name, estimators, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        # evaluate over sampled dataset
        evaluate_estimators(file_name, estimators, X_train=X_train_resampled,  X_test=X_test,  y_train=y_train_resampled, y_test=y_test)
        
        with open(file_name + ".txt", "a") as file:
            for estimator in estimators:
                file.write(f'================== {estimator["model"]["name"]} ======================\n')
                for key in estimator["metrics"]:
                    file.write(f'{key}: {estimator["metrics"][key]} \n')

                file.write(f"\n")




