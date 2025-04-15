'''
Author: Group AlphaML
March 21, 2025
Description: Extract the column correlation and returns the list of strongest correlated features.
             Now includes enhanced EDA for class imbalance and feature distributions.
'''
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Filters correlation matrix to only keep relationships within defined thresholds
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
    # CHANGE: Updated save path
    plt.savefig(f"results/eda/{filename}", dpi=300)
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
    # CHANGE: Updated save path
    plt.savefig(f"results/eda/{filename}.png", dpi=300)
    plt.close()

    return top_corrs

# Saves correlation summary to .txt file
def save_correlation_summary(corr_df, filename):
    # CHANGE: Updated save path
    with open(f"results/eda/{filename}.txt", "w", encoding="utf-8") as f:
        f.write("Top Positive and Negative Correlations:\n\n")
        for _, row in corr_df.iterrows():
            f.write(f"{row['Feature 1']} <-> {row['Feature 2']}: {row['Correlation']:+.3f}\n")

# CHANGE: Added function for class imbalance analysis (per professor's feedback on EDA)
def analyze_class_imbalance(df, target_col, year):
    plt.figure(figsize=(10, 6))
    class_counts = df[target_col].value_counts()
    ax = sns.barplot(x=class_counts.index, y=class_counts.values)
    
    plt.title(f'Class Distribution for {target_col} - {year} Dataset')
    plt.xlabel(target_col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f"results/eda/class_imbalance_{year}.png", dpi=300)
    plt.close()

# CHANGE: Added function for feature distribution analysis (per professor's feedback on EDA)
def analyze_feature_distributions(df, year):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:6]  # Limit to avoid too many plots
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols):
        if i < 6:  # Limit to 6 plots
            plt.subplot(2, 3, i + 1)
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
    
    plt.tight_layout()
    plt.savefig(f"results/eda/feature_distributions_{year}.png", dpi=300)
    plt.close()


def correlation(df, year, pre_norm_df=None):
    """
    Analyze correlations and create EDA visualizations
    
    Parameters:
        df: Normalized dataframe for correlation analysis
        year: Dataset year for labeling
        pre_norm_df: Optional pre-normalized dataframe for feature distribution plots
    """
    # CHANGE: Use pre-normalized data for distribution analysis if provided
    data_for_eda = pre_norm_df if pre_norm_df is not None else df
    
    # CHANGE: Added class imbalance analysis (per professor's feedback)
    if 'HadHeartAttack' in df.columns:
        analyze_class_imbalance(df, 'HadHeartAttack', year)
    elif 'HeartDiseaseorAttack' in df.columns:
        analyze_class_imbalance(df, 'HeartDiseaseorAttack', year)
    
    # CHANGE: Use pre_norm_df for feature distributions if available
    analyze_feature_distributions(data_for_eda, year)
    
    # Correlation matrices
    corr = df.corr(numeric_only=True)

    # Filtered for readability
    filtered = filter_correlation_matrix(corr)

    # Saves heatmaps
    plot_and_save_heatmap(filtered, f"Expanded Correlation Matrix - {year} Dataset (|r| > 0.15)", f"correlation_matrix_{year}.png")

    # Saves infographics and summaries
    top_corrs = plot_extreme_correlations(corr, f'Top Feature Relationships - {year} Dataset', f"correlation_infographic_{year}")

    # list of most related features: Concatenate Feature 1 and Feature 2 into a single column with unique values
    top_features = np.unique(pd.concat([top_corrs['Feature 1'], top_corrs['Feature 2']]).to_numpy())

    save_correlation_summary(top_corrs, f"correlation_summary_{year}")
    return top_features