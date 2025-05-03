'''
Author: Group AlphaML
April 17, 2025
Description: Runs Random Forest model on the datasets.
Dataset Sources:
1. 2015 Dataset: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
2. 2022 Dataset: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
'''

import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Defining random forest classifier function
def random_forest_classifier(X, y, dataset_name):
    """
    Train a Random Forest classifier with hyperparameter tuning and evaluate its performance.

    Parameters
    ----------
    X : array-like
        Feature matrix for training.
    y : array-like
        Target labels for training.
    dataset_name : str
        Name of the dataset, used for saving results.

    Returns
    -------
    rf_best_model : RandomForestClassifier
        The best estimator found by GridSearchCV.
    """

    start = time.time()

    rf_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Random state for reproducibility

    rf_model = RandomForestClassifier(random_state=42) # Random state for reproducibility

    rf_param_grid = {
        'n_estimators': [200],
        'max_depth': [25],
        'min_samples_split': [5],
        'min_samples_leaf': [3],
        'class_weight': [{0: 1, 1: 11}],
        'max_features': ['sqrt'],
        'bootstrap': [False]
    }

    rf_grid = GridSearchCV(
        estimator=rf_model,
        param_grid=rf_param_grid,
        cv=rf_skf,
        scoring='f1',
        n_jobs=8,
        verbose=1
    ) #n_jobs=8 for parallel processing

    rf_grid.fit(X, y)

    rf_best_model = rf_grid.best_estimator_
    rf_y_pred = rf_best_model.predict(X)
    rf_y_proba = rf_best_model.predict_proba(X)[:, 1]

    end = time.time()

    # Save classification report
    rf_report = classification_report(y, rf_y_pred, digits=4, output_dict=True)
    with open(f"results/model_performance/rf_report_{dataset_name}.txt", "w") as f:
        f.write(f"Random Forest Report ({dataset_name})\n\n")
        for label, metrics in rf_report.items():
            f.write(f"{label}:\n")
            if isinstance(metrics, dict):
                for m, val in metrics.items():
                    f.write(f"  {m}: {val:.4f}\n")
            else:
                f.write(f"  {metrics:.4f}\n")
        f.write(f"\nBest Params: {rf_grid.best_params_}\n")
        f.write(f"Runtime (s): {end - start:.2f}\n")

    # Confusion Matrix
    rf_cm = confusion_matrix(y, rf_y_pred)
    plt.figure()
    sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"results/model_performance/rf_confusion_matrix_{dataset_name}.png", dpi=300)
    plt.close()

    # ROC Curve
    rf_fpr, rf_tpr, _ = roc_curve(y, rf_y_proba)
    rf_roc_auc = auc(rf_fpr, rf_tpr)
    plt.figure()
    plt.plot(rf_fpr, rf_tpr, label=f"AUC = {rf_roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"results/model_performance/rf_roc_curve_{dataset_name}.png", dpi=300)
    plt.close()

    return rf_best_model