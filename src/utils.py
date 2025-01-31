

def clean_dataframe(df):
    """
    Cleans specified columns in a DataFrame by removing commas, dots, and newlines.

    Parameters:
    df (pd.DataFrame): The DataFrame to clean.
    columns (list): List of column names to clean.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    columns = [col for col in df.columns.to_list() if col != 'verse']
    df_cleaned = df.copy()
    df_cleaned[columns] = (
        df_cleaned[columns]
        .replace({",": "", "\.": "", "\n": " "}, regex=True)  # Remove unwanted characters
        .apply(lambda x: x.str.lower())  # Convert to lowercase
    )
    return df_cleaned