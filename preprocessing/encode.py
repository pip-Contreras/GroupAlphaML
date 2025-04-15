'''
Author: Group AlphaML
March 21, 2025
Description: Classifies features en encodes them using OneHotEncoder or LabelEncoder as necessary. 
'''

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def encode(dataset, columns):
    """
    Encodes the specified columns using OneHotEncoder or LabelEncoder based on the rules defined.
    
    Parameters:
        dataset (pd.DataFrame): The input dataset.
        columns (list): The list of columns to encode.
    
    Returns:
        pd.DataFrame: The fully encoded dataset.
    """
    encoded_dataset = dataset.copy()
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    label_encoder = LabelEncoder()

    for column in columns:
        if column in dataset.columns:
            # Apply OneHotEncoder for nominal categorical data
            # CHANGE: Removed GenHlth from this list to fix inconsistency (per professor's feedback)
            if column in ['Age', 'DiffWalk', 'PhysActivity']:
                encoded_values = one_hot_encoder.fit_transform(dataset[[column]])
                one_hot_labels = one_hot_encoder.get_feature_names_out([column])
                one_hot_df = pd.DataFrame(encoded_values, columns=one_hot_labels, index=dataset.index)
                encoded_dataset = pd.concat([encoded_dataset, one_hot_df], axis=1)
                encoded_dataset.drop(column, axis=1, inplace=True)
            
            # Apply LabelEncoder for ordinal categorical data
            elif column in ['Education', 'Income', 'GenHlth', 'MentHlth', 'PhysHlth']:
                encoded_dataset[column] = label_encoder.fit_transform(dataset[column])
            
            # Binary encoding for `HadHeartAttack`
            elif column == 'HadHeartAttack':
                encoded_dataset[column] = dataset[column].map({'No': 0, 'Yes': 1})

            # Leave numerical or binary columns as they are
            elif column in ['BMI', 'HeightInMeters', 'MentalHealthDays', 'PhysicalHealthDays',
                            'SleepHours', 'WeightInKilograms', 'HighBP', 'HeartDiseaseorAttack']:
                pass  # No encoding needed
        else:
            print(f"Column {column} not found in the dataset.")
    
    return encoded_dataset


"""
Column Name,Encoding Type, Reasoning
BMI,None,Continuous numerical
HeightInMeters,None,Continuous numerical
MentalHealthDays,None,Continuous numerical
PhysicalHealthDays,None,Continuous numerical
SleepHours,None,Continuous numerical
WeightInKilograms,None,Continuous numerical
Age,OneHotEncoder,Nominal categorical
DiffWalk,OneHotEncoder,Nominal categorical
Education,LabelEncoder,Ordinal categorical
GenHlth,LabelEncoder,Ordinal categorical (order matters)
HighBP,None,Binary numerical (0/1)
Income,LabelEncoder,Ordinal categorical (order matters)
MentHlth,LabelEncoder,Ordinal categorical (order matters)
PhysActivity,OneHotEncoder,Nominal categorical
PhysHlth,LabelEncoder,Ordinal categorical (order matters)

HadHeartAttack, Binary Numerical, only two values
HeartDiseaseorAttack, None, already binary
"""