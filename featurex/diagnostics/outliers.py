'''
Diagnostic functions for detecting outliers in the data
'''

import pandas as pd
import numpy as np

from scipy.spatial.distance import mahalanobis

def mahalanobis_distances(df, axis=0):
    '''
    Returns a pandas Series with Mahalanobis distances for each sample on the axis.

    Args:
        df: pandas DataFrame with columns to run diagnostics on
        axis: 0 to find outlier rows, 1 to find outlier columns
    '''
    df = df.transpose() if axis == 1 else df
    means = df.mean()
    inv_cov = np.linalg.inv(df.cov())
    dists = []
    for i, sample in df.iterrows():
        dists.append(mahalanobis(sample, means, inv_cov))

    return pd.Series(dists, df.index, name='Mahalanobis')
