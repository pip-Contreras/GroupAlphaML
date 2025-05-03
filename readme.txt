# GroupAlphaML

This project applies machine learning techniques to predict heart disease risk using two large datasets from different time periods (2015 and 2022). We preprocess the data, explore it with visualizations, and implement multiple classification algorithms to evaluate performance across models.

-----------

## Requirements

- Python 3.10 or higher installed.
- Install dependencies with:

pip install -r requirements.txt

-----------

## How to Run the program

1. Prepare the environment
	- Install VS Code with Python and Jupyter extensions.
2. Run the script
	- Open a terminal at the project directory
	- Run the main script with command: `python main.py`


This script coordinates data preprocessing and model training, saving outputs to the `/results` folder.

-----------

## Project Structure

/project_root
├── models/
│   ├── XGB_MLP.py
│   ├── logistic_regression.py
│   ├── mlp.py
│   └── random_forest.py
├── preprocessing/
│   ├── cleaning.py
│   ├── correlation.py
│   ├── encode.py
│   └── normalization.py
├── processed_data/
│   └── structure.txt
├── raw_data/
│   └── structure.txt
├── results/
│   ├── eda/
│   │   ├── class_imbalance_2015.png
│   │   ├── class_imbalance_2022.png
│   │   ├── correlation_inforgraphic_2015.png
│   │   ├── correlation_inforgraphic_2022.png
│   │   ├── correlation_matrix_2015.png
│   │   ├── correlation_matrix_2022.png
│   │   ├── correlation_summary_2015.txt
│   │   ├── correlation_summary_2022.txt
│   │   ├── feature_distributions_2015.png
│   │   └── feature_distributions_2022.png
│   └── model_performance/
│       ├── [multiple report and plot files for each model & dataset]
├── Phase3_Change_Log_and_Planning.txt
├── main.py
├── readme.txt
└── requirements.txt

-----------

## Codebase Overview

`main.py` - The main entry point that links preprocessing, training, and results.
`models/` - Contains modular scripts for each machine learning model.
`preprocessing/` - Scripts for cleaning, encoding, scaling, and analyzing the data.
`results/` - All generated plots (EDA + model performance) and report files.
`raw_data/ & processed_data/` - For raw input data and cleaned/processed data storage.

-----------

## Dataset Information

1.  2022 Dataset:

- Name: heart_2022_with_nans.csv
- Source: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
- Description: A dataset containing health indicators and heart disease risk factors collected in 2022.

2.  2015 Dataset:

- Name: heart_disease_health_indicators_BRFSS2015.csv
- Source: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
- Description: Health indicators and behavioral risk factors from the 2015 Behavioral Risk Factor Surveillance System (BRFSS).

-----------

## Output Information

All plots, metrics, and reports, are saved inside the results folder.

EDA (Exploratory Data Analysis):

	- Class imbalance plots
	- Correlation matrices + summaries
	- Feature distribution plots

Model Performance:

	- ROC Curves
	- Confusion matrices
	- Classification reports (.txt files)

-----------

## Libraries and Versions

The following Python packages are required to run the project (matching requirements.txt):

Core Libraries:

- numpy==2.2.2
- pandas==2.2.3
- scipy==1.15.1
- python-dateutil==2.9.0.post0
- pytz==2024.2
- tzdata==2025.1

Machine Learning:

- scikit-learn==1.6.1
- ucimlrepo==0.0.7
- joblib==1.4.2
- threadpoolctl==3.5.0

Plotting & Visualization:

- matplotlib==3.10.0
- seaborn==0.13.2
- contourpy==1.3.1
- cycler==0.12.1
- fonttools==4.55.5
- kiwisolver==1.4.8
- pillow==11.1.0
- pyparsing==3.2.1
- packaging==24.2

Other Dependencies:

- certifi==2025.1.31
- six==1.17.0

You can also confirm exact versions by running command:

pip freeze > requirements.txt

-----------

## Additional Notes

- The program automatically creates necessary folders (like `/results/model_performance/`) if missing.
- Each model is modular and can be run independently from the `models/` folder if desired (e.g., `python models/logistic_regression.py`).
