'''
Author: Group AlphaML
March 21, 2025
Description: Preprocessing for heart attack datasets.
Dataset Sources:
1. 2015 Dataset: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
2. 2022 Dataset: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.cleaning import cleaning
from preprocessing.correlation import correlation
from preprocessing.encode import encode
from preprocessing.normalization import normalization

def save_to_csv(dataset, filename):
    dataset.to_csv(filename, index=False) 
    print(f"Dataset saved as {filename}")


def main():
    # loading files
    print("Loading raw datasets...")
    # CHANGE: Updated file paths to use raw_data directory
    df_2022 = pd.read_csv("raw_data/heart_2022_with_nans.csv")
    df_2015 = pd.read_csv("raw_data/heart_disease_health_indicators_BRFSS2015.csv")

    # cleaning
    print("Cleaning datasets...")
    df_2022 = cleaning(df_2022)
    df_2015 = cleaning(df_2015)

    # encoding
    print("Encoding categorical features...")
    df_2022 = encode(df_2022, df_2022.columns)
    df_2015 = encode(df_2015, df_2015.columns)
    
    # CHANGE: Make pre-normalized copies AFTER encoding is complete
    # Make deep copies to ensure they're completely separate objects
    print("Creating pre-normalized copies...")
    df_2022_pre_norm = df_2022.copy(deep=True)
    df_2015_pre_norm = df_2015.copy(deep=True)
    
    # Verify that copies are made with a simple output
    print(f"2022 pre-norm dataset shape: {df_2022_pre_norm.shape}")
    print(f"2015 pre-norm dataset shape: {df_2015_pre_norm.shape}")

    # normalization
    print("Normalizing datasets...")
    df_2022 = normalization(df_2022)
    df_2015 = normalization(df_2015)

    # CHANGE: Run correlation on processed data (to get more accurate results)
    # CHANGE: Pass both normalized and pre-normalized data to correlation function
    print("Performing correlation analysis and EDA on processed data...")
    columns_2022 = correlation(df_2022, "2022", pre_norm_df=df_2022_pre_norm)
    columns_2015 = correlation(df_2015, "2015", pre_norm_df=df_2015_pre_norm)
    
    # add target column to the list (if not already included)
    if "HadHeartAttack" not in columns_2022:
        columns_2022 = np.append(columns_2022, "HadHeartAttack")
    if "HeartDiseaseorAttack" not in columns_2015:
        columns_2015 = np.append(columns_2015, "HeartDiseaseorAttack")

    # Filter to important features
    # Only keep columns that actually exist in the dataframe after processing
    columns_2022 = [col for col in columns_2022 if col in df_2022.columns]
    columns_2015 = [col for col in columns_2015 if col in df_2015.columns]
    
    df_2015 = df_2015[columns_2015]
    df_2022 = df_2022[columns_2022]

    # store
    # CHANGE: Updated file paths to use processed_data directory
    print("Saving processed datasets...")
    save_to_csv(df_2022, "processed_data/df_2022.csv")
    save_to_csv(df_2015, "processed_data/df_2015.csv")
    
    # CHANGE: Also save the pre-normalized data for reference
    save_to_csv(df_2022_pre_norm, "processed_data/df_2022_pre_norm.csv")
    save_to_csv(df_2015_pre_norm, "processed_data/df_2015_pre_norm.csv")
    
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()