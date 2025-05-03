'''
Author: Group AlphaML
March 21, 2025
Description: Cleans dataset from null values
'''
def cleaning(df):    
    """
    Cleans a DataFrame by imputing missing values and removing any remaining incomplete rows.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing missing values.

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame with missing values imputed or removed.
    """

    # CHANGE: Create a copy to avoid modifying the original DataFram just in case we meention it elsewhere
    df = df.copy()

    # CHANGE: Print missing values before cleaning
    print("Missing values before cleaning:")
    print(df.isnull().sum())

    # Extract and show missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    # For numerical columns, use median
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # For categorical columns, use mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # CHANGE: Record row count before dropna
    before_drop = df.shape[0]

    # Drop any remaining rows with missing values (should be minimal after imputation)
    df.dropna(inplace=True)

    # CHANGE: Report how many rows were dropped
    after_drop = df.shape[0]
    print(f"Rows dropped after imputation: {before_drop - after_drop}")

    # CHANGE: Print missing values after cleaning
    print("Missing values after cleaning:")
    print(df.isnull().sum())

    return df
