def variances(df):
    '''
    Returns a pandas Series with variances for each column

    Args:
        df: pandas DataFrame with columns to run diagnostics on
    '''
    return df.var(axis=0)