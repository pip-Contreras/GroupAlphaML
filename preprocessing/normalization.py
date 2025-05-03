'''
Author: Group AlphaML
March 21, 2025
Description: Normalizes the given dataset with MinMax scaler.
'''
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

def plot(df):
    """
    Generates pairwise scatter plots for all features in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing numerical and categorical features.
    """
    sns.pairplot(df)


def normalization(df):
    """
    Normalizes numeric features using Min-Max scaling while preserving non-numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing numerical and categorical features.

    Returns
    -------
    pd.DataFrame
        A DataFrame with normalized numeric features and unchanged categorical features.
    """

    # CHANGE: Keep only numeric columns for scaling
    numeric_cols = df.select_dtypes(include=['float64', 'int64', 'int32', 'float32']).columns
    df_numeric = df[numeric_cols]

    # CHANGE: Check for constant columns before normalization (per professor's feedback)
    const_cols = [col for col in df_numeric.columns if df_numeric[col].nunique() <= 1]
    if const_cols:
        df_numeric = df_numeric.drop(columns=const_cols)

    scaler = MinMaxScaler()
    normalized_df = scaler.fit_transform(df_numeric)
    normalized_df = pd.DataFrame(normalized_df, columns=df_numeric.columns, index=df.index)

    # CHANGE: Recombine with unscaled non-numeric columns
    non_numeric = df.drop(columns=numeric_cols).copy()
    result_df = pd.concat([normalized_df, non_numeric], axis=1)

    # CHANGE: Check for NaN values after normalization (per professor's feedback)
    if result_df.isnull().sum().sum() > 0:
        result_df = result_df.fillna(0)

    return result_df
