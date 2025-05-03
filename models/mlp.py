'''
Author: Group AlphaML
April 17, 2025
Description: Runs Multilayer Perceptron model on the datasets.
Dataset Sources:
1. 2015 Dataset: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
2. 2022 Dataset: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
'''

from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import time
from imblearn.over_sampling import SMOTE
import numpy as np


# =============================================================================
# Data
# =============================================================================
def get_data():
    """Loads two datasets into json divided into features and labels."""
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
    """Oversamples positive class using SMOTE.
    
    Parameters
    ----------
    X_train : array-like
        Training feature set.
    y_train : array-like
        Training target labels.
    """
    smote = SMOTE(random_state=1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def save_class_ratios(y, file_name, title):
    """
    Compute class distribution ratios and save them to a file.

    Parameters
    ----------
    y : array-like
        Array of class labels for the dataset.
    file_name : str
        Name of the file where class ratios will be saved (without extension).
    title : str
        Title to be written in the file for clarity.
    """
    print(file_name)
    print(title)
    classes, counts = np.unique(y, return_counts=True)
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
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score: .2f})')
    plt.plot([0,1], [0,1], linestyle="--", color="gray")  # ref line

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(filename, dpi=300)


# =============================================================================
# Training Functions
# =============================================================================
def get_estimators(): # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    """Returns a JSON for different models for CVSearch."""
    estimators = [
        {
            "model": {
                "name": "MLP",
                "estimator": MLPClassifier(max_iter=500, random_state=1, early_stopping=True),
            },
            "params": {
                "hidden_layer_sizes": [(50,), (100,), (100, 50)],
                "activation": ["relu", "tanh"],
                "learning_rate_init": [0.001, 0.01],
                "alpha": [0.0001, 0.001, 0.01]
            },
            "best_model": [],
            "metrics": {
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
    """
    Train an estimator, evaluate performance metrics, and plot ROC curve.

    Parameters
    ----------
    file_name : str
        The base file name for saving the ROC curve plot.
    estimator : dict
        Dictionary containing:
        - "model": A dictionary with "estimator" (a scikit-learn model) and "name" (model identifier).
        - "params": Dictionary of hyperparameters.
    X_train : array-like
        Training feature set.
    X_test : array-like
        Test feature set.
    y_train : array-like
        Training target labels.
    y_test : array-like
        Test target labels.
    """
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
    """
    Perform hyperparameter tuning and evaluation for multiple configurations 
    of MLP using stratified k-fold cross-validation.

    Parameters
    ----------
    file_name : str
        The name of the file where evaluation metrics will be stored.
    estimators : list
        A list of dictionaries, each containing:
        - "model": A dictionary with the key "estimator" (a scikit-learn model).
        - "params": Dictionary of hyperparameters for grid search.
    X_train : array-like
        Training feature set.
    X_test : array-like
        Test feature set.
    y_train : array-like
        Training target labels.
    y_test : array-like
        Test target labels.
    """
    # Stratified k-fold cross-validation is good for unbalanced datasets
    n_splits = 5  # 5 is a good balance for medium size datasets.
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

    try:
        for estimator in estimators:
            grid_search = HalvingGridSearchCV(
                estimator=estimator["model"]["estimator"],
                param_grid=estimator["params"],
                cv=stratified_kfold,
                factor=2,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            # train estimator
            grid_search.fit(X_train, y_train)
            # pick best
            estimator["best_model"] = grid_search.best_estimator_
            estimator["model"]["estimator"] = grid_search.best_estimator_  # update for evaluation
            # evaluate
            set_metrics(file_name, estimator, X_train, X_test, y_train, y_test)
    except KeyboardInterrupt:
        exit()
    except Exception as e:
        print("Error optimizing model: ", e)


def main():
    """
    Train and evaluate multiple estimators on different datasets.

    This function:
    - Splits datasets into training and test sets using stratified sampling.
    - Applies oversampling to handle class imbalance.
    - Evaluates models using both original and oversampled datasets.
    - Saves class distribution ratios and model performance metrics to files.
    """
    for dataset in get_data():
        # splitting datasets
        X_train, X_test, y_train, y_test = train_test_split(
            dataset["X"], dataset["y"],
            test_size=0.2,
            random_state=1,
            stratify=dataset["y"]
        )
        
        # oversampling
        X_train_resampled, y_train_resampled = oversample_class(X_train, y_train)  
        estimators = get_estimators()

        # Save dataset class distribution
        file_name = f"results/model_performance/{dataset['name']}"
        title = f"Training Dataset: Original dataset (80%)"
        title_oversampled = f"Training Dataset: Oversampled dataset (80%)"
        save_class_ratios(y_train, file_name, title)
        save_class_ratios(y_train_resampled, file_name, title_oversampled)

        # evaluate original dataset
        evaluate_estimators(file_name, estimators, X_train, X_test, y_train, y_test)
        # evaluate oversampled dataset
        evaluate_estimators(file_name, estimators, X_train_resampled, X_test, y_train_resampled, y_test)
        
        with open(file_name + ".txt", "a") as file:
            for estimator in estimators:
                file.write(f'================== {estimator["model"]["name"]} ======================\n')
                for key in estimator["metrics"]:
                    file.write(f'{key}: {estimator["metrics"][key]} \n')

            # log best parameters
                if estimator["best_model"]:
                    file.write("Best Parameters:\n")
                    for param, val in estimator["best_model"].get_params().items():
                        if param in estimator["params"]:
                            file.write(f"  {param}: {val}\n")
                file.write("\n")


if __name__ == "__main__":
    """Script entry point for standalone run."""
    main()

