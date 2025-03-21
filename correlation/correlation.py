# ------------------------------------------------------------
# File: correlation.py
# Course: C S-519-M70 - Applied Machine Learning I
# Group: Alpha
# To Run: Python correlation/correlation.py
# Description:
# This script performs correlation analysis on two datasets 
# related to heart disease classification and generates:
# - Correlation heatmap matrices
# - Infographic-style plots of the strongest correlations (positive and negative)
# - .txt summaries of the top positive and negative feature relationships
#
# These analyses can be used to support feature selection and interpretation
# in preparation for machine learning modeling.
#
# Dataset Sources:
# 1. 2015 Dataset: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
# 2. 2022 Dataset: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
# ------------------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# Makes sure that the charts folder exists
os.makedirs("correlation/charts", exist_ok=True)

# Loads datasets
df_2022 = pd.read_csv("correlation/correlation_data/heart_2022_with_nans.csv")
df_2015 = pd.read_csv("correlation/correlation_data/heart_disease_health_indicators_BRFSS2015.csv")

# This converts Yes/No to 1/0 for correlation analysis
def binarize_columns(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object and df[col].dropna().isin(['Yes', 'No']).all():
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    return df

# Filters correlation matrix to only keep relationships within defined thresholds to show features with stronger correlations
def filter_correlation_matrix(corr_matrix, threshold=0.15):
    mask = (corr_matrix.abs() > threshold) & (corr_matrix != 1.0)
    rows_to_keep = mask.any(axis=1)
    cols_to_keep = mask.any(axis=0)
    return corr_matrix.loc[rows_to_keep, cols_to_keep]

# Plots and saves correlation heatmaps
def plot_and_save_heatmap(corr_matrix, title, filename):
    plt.figure(figsize=(18, 14))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True,
                annot_kws={"size": 10}, cbar_kws={"shrink": 0.8})
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"correlation/charts/{filename}", dpi=300)
    plt.close()

# Plots strongest positive and negative correlations in an easy to interpret style
def plot_extreme_correlations(corr_matrix, title, filename, top_n=6):
    corr_pairs = corr_matrix.unstack().reset_index()
    corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
    corr_pairs = corr_pairs[corr_pairs['Feature 1'] != corr_pairs['Feature 2']]
    corr_pairs = corr_pairs.drop_duplicates(subset=['Correlation'])

    top_positive = corr_pairs.sort_values(by='Correlation', ascending=False).head(top_n)
    top_negative = corr_pairs.sort_values(by='Correlation', ascending=True).head(top_n)
    top_corrs = pd.concat([top_positive, top_negative])

    plt.figure(figsize=(12, 8))
    y_pos = list(range(len(top_corrs)))
    colors = ['green' if val > 0 else 'red' for val in top_corrs['Correlation']]

    for i, (f1, f2, corr) in enumerate(top_corrs.values):
        label = f"{f1} → {f2}"
        plt.arrow(0, i, corr, 0, color=colors[i], width=0.05, head_width=0.2, head_length=0.05)
        plt.text(corr + (0.02 if corr > 0 else -0.02), i, f"{corr:+.2f}", va='center',
                 ha='left' if corr > 0 else 'right', fontsize=10)
        plt.text(-1.05, i, label, va='center', ha='left', fontsize=11)

    plt.yticks([])
    plt.xticks([-1, -0.5, 0, 0.5, 1])
    plt.title(title, fontsize=14)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)

    green_patch = mpatches.Patch(color='green', label='Positive Correlation')
    red_patch = mpatches.Patch(color='red', label='Negative Correlation')
    plt.legend(handles=[green_patch, red_patch], loc='lower right')

    plt.xlim(-1.2, 1.2)
    plt.tight_layout()
    plt.savefig(f"correlation/charts/{filename}.png", dpi=300)
    plt.close()

    return top_corrs

# Saves correlation summary to .txt file
def save_correlation_summary(corr_df, filename):
    with open(f"correlation/charts/{filename}.txt", "w", encoding="utf-8") as f:
        f.write("Top Positive and Negative Correlations:\n\n")
        for _, row in corr_df.iterrows():
            f.write(f"{row['Feature 1']} <-> {row['Feature 2']}: {row['Correlation']:+.3f}\n")

# Preps data
df_2022_clean = binarize_columns(df_2022).select_dtypes(include='number').dropna()
df_2015_clean = binarize_columns(df_2015).select_dtypes(include='number').dropna()

# Correlation matrices
corr_2022 = df_2022_clean.corr(numeric_only=True)
corr_2015 = df_2015_clean.corr(numeric_only=True)

# Filtered for readability
filtered_2022 = filter_correlation_matrix(corr_2022)
filtered_2015 = filter_correlation_matrix(corr_2015)

# Saves heatmaps
plot_and_save_heatmap(filtered_2022, "Expanded Correlation Matrix - 2022 Dataset (|r| > 0.15)", "correlation_matrix_2022.png")
plot_and_save_heatmap(filtered_2015, "Expanded Correlation Matrix - 2015 Dataset (|r| > 0.15)", "correlation_matrix_2015.png")

# Saves infographics and summaries
top_corrs_2022 = plot_extreme_correlations(corr_2022, "Top Feature Relationships - 2022 Dataset", "correlation_infographic_2022")
top_corrs_2015 = plot_extreme_correlations(corr_2015, "Top Feature Relationships - 2015 Dataset", "correlation_infographic_2015")

save_correlation_summary(top_corrs_2022, "correlation_summary_2022")
save_correlation_summary(top_corrs_2015, "correlation_summary_2015")
