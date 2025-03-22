###############################################################################################################################################
# Author: Jonas Engstrom - Group ALPHA
# Date: 3/21/2025
# Purpose: The purpose of this script is to endcode the labels of the two datasets being used ('heart_2022_with_nans.csv' and 'heart_disease_health_indicators_BRFSS2015.csv') 
#          for our heart disease classification project in order enable proper mulit-class classifcation or any other method
#          we plan to implement.
# Common columns in both datasets to be used to cross reference:
#                                                               2022:                             2015:
#                                                                    -HeartDiseaseorAttack(binary)     -HadHeartAttack(binary)
#                                                                    -BMI(numerical)                   -BMI(numerical)              
#                                                                    -PhysicalActivities(binary)       -PhysActivity(binary)
#                                                                    -GeneralHealth(categorical)       -GenHlth(categorical)
#                                                                    -Sex(binary)                      -Sex(binary)
#                                                                    -AgeCategory(categorical)         -Age(categorical)
#                                                                    -DifficultyWalking(binary)        -DiffWalk(binary)
#                                                                    -MentalHealthDays(numerical)      -MentHlth(numerical)
#                                                                    -PhysicalHealthDays(numerical)    -PhysHlth(numerical)

# Unique columns to each dataset or different categorization:
#                                                               2022:                             2015:
#                                                                    -State(categorical)               -HighBP(binary)
#                                                                    -LastCheckupTime(categorical)     -HighChol(binary)
#                                                                    -SleepHours(numerical)            -CholCheck(binary)
#                                                                    -RemovedTeeth(categorical)        -Fruits(binary)
#                                                                    -HadAngina(binary)                -Veggies(binary)
#                                                                    -HadAsthma(binary)                -HvyAlcoholConsump(binary)
#                                                                    -HadSkinCancer(binary)            -AnyHealthcare(binary)
#                                                                    -HadCOPD(binary)                  -NoDocbcCost(binary)
#                                                                    -HadStroke(binary)                -Education(categorical)
#                                                                    -HadDepressiveDisorder(binary)    -Income(categorical)
#                                                                    -HadKidneyDisease(binary)         -Stroke(binary)
#                                                                    -HadArthritis(binary)             -Diabetes(categorical)             
#                                                                    -DeafOrHardOfHearing(binary)      -Smoker(binary)
#                                                                    -BlindOrVisionDiificulty(binary)
#                                                                    -DifficultyConcentrating(binary)
#                                                                    -DifficultyDressingBathing(binary)
#                                                                    -DifficultyErrands(binary)
#                                                                    -ECigaretteUsage(categorical)
#                                                                    -ChestScan(binary)
#                                                                    -RaceEthnicityCategory(categorical)
#                                                                    -HeightInMeters(numerical)
#                                                                    -WeightInKilograms(numerical)
#                                                                    -AlcoholDrinkers(binary)
#                                                                    -HIVTesting(binary)
#                                                                    -FluVaxLast12(binary)
#                                                                    -PneumoVaxEver(binary)
#                                                                    -TetanusLast10Tdap(categorical)
#                                                                    -HighRiskLastYear(binary)
#                                                                    -CovidPos(binary)
#                                                                    -HadDiabetes(binary)
#                                                                    -SmokerStatus(categorical)  
# Future updates:
#                -Adjust column names to match similar columns in each dataset for simplified training.
#                - Utilize the codebooks to ensure accuracy of encoding and categorization to maintain consistency across both datasets.                                                                         
###############################################################################################################################################
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load datasets
df2022 = pd.read_csv('C:/Users/jceng/Documents/CS_519/datasets/heart_2022_no_nans.csv')
df2015 = pd.read_csv('C:/Users/jceng/Documents/CS_519/datasets/heart_disease_health_indicators_BRFSS2015.csv')

# 2022 encoding
# Binary mapping for colu8mns that use "Yes"/"No"
binary_map = {'Yes': 1, 'No': 0}
binary_cols_2022 = [
    'PhysicalActivities', 'HadHeartAttack', 'HadAngina', 'HadStroke',
    'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder',
    'HadKidneyDisease', 'HadArthritis', 'HadDiabetes', 'DeafOrHardOfHearing',
    'BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyWalking',
    'DifficultyDressingBathing', 'DifficultyErrands', 'ChestScan',
    'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver',
    'HighRiskLastYear', 'CovidPos'
]

for col in binary_cols_2022:
    if col in df2022.columns:
        df2022[col] = df2022[col].map(binary_map)

# Male and female mapping to 1 and 0 respectively
if 'Sex' in df2022.columns:
    sex_map = {'Female': 0, 'Male': 1}
    df2022['Sex'] = df2022['Sex'].map(sex_map)

# One0hot encode the columns that have text
cat_cols_2022 = [
    'GeneralHealth', 'LastCheckupTime', 'RemovedTeeth',
    'SmokerStatus', 'ECigaretteUsage', 'RaceEthnicityCategory',
    'AgeCategory', 'TetanusLast10Tdap'
]

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit and transform categorical columns
ohe_array = ohe.fit_transform(df2022[cat_cols_2022])
ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out(cat_cols_2022), index=df2022.index)

# Drop original categorical columns and merge one-hot encoded columns
df2022_encoded = pd.concat([df2022.drop(columns=cat_cols_2022), ohe_df], axis=1)

# 2015 encoding
# All columns in 2015 dataset are numeric(binary or ordinal)
# Code below for future use.
'''cat_cols_2015 = [col for col in df2015.columns if df2015[col].dtype == 'object']

if cat_cols_2015:
    ohe2015 = OneHotEncoder(sparse=False, handle_unknown='ignore')
    ohe_array_2015 = ohe2015.fit_transform(df2015[cat_cols_2015])
    ohe_df_2015 = pd.DataFrame(ohe_array_2015, columns=ohe2015.get_feature_names_out(cat_cols_2015), index=df2015.index)
    df2015_encoded = pd.concat([df2015.drop(columns=cat_cols_2015), ohe_df_2015], axis=1)
else:
    df2015_encoded = df2015.copy()
'''
# Save encoded datasets to specified path
df2022_encoded.to_csv('C:/Users/jceng/Documents/CS_519/encoded_datasets/encoded_heart_2022.csv', index=False)
#df2015_encoded.to_csv('C:/Users/jceng/Documents/CS_519/encoded_datasets/encoded_heart_2015.csv', index=False)

print("Finished encoding.")
