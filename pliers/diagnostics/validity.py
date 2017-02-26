'''
Diagnostic functions for detecting validity of features
'''

import pandas as pd


def variances(df):
    '''
    Returns a pandas Series with variances for each column

    Args:
        df: pandas DataFrame with columns to run diagnostics on
    '''
    return pd.Series(df.var(axis=0), name='Variances')
