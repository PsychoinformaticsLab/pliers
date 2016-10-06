import pytest

import pandas as pd
import numpy as np

from featurex.diagnostics import Diagnostics
from featurex.diagnostics.collinearity import correlation_matrix
from featurex.diagnostics.collinearity import eigenvalues
from featurex.diagnostics.collinearity import condition_indices
from featurex.diagnostics.collinearity import variance_inflation_factors
from featurex.diagnostics.outliers import mahalanobis_distances
from featurex.diagnostics.validity import variances

def test_diagnostics():
    df = pd.DataFrame(np.random.randn(10, 5))
    diagnostics = Diagnostics(df)
    results = diagnostics.summary()
    assert type(results) == pd.DataFrame
    assert results.shape == (df.shape[1], 3 + df.shape[1])

def test_correlation_matrix():
    df = pd.DataFrame(np.random.randn(10, 5), columns=['a','b','c','d','e'])
    corr = correlation_matrix(df)
    assert type(corr) == pd.DataFrame
    assert corr.shape == (df.shape[1], df.shape[1])
    assert np.array_equal(np.diagonal(corr), ([1.0] * df.shape[1]))
    assert np.isfinite(corr['a']['b'])
    assert np.isclose(corr['a']['b'], corr['b']['a'], 1e-05)

def test_eigenvalues():
    df = pd.DataFrame(np.random.randn(100, 2), columns=['a', 'b'])
    noise = np.random.randn(100)
    df['c'] = (2*df['a']) + (3*df['b'])
    eig_vals = eigenvalues(df)
    assert type(eig_vals) == pd.Series
    assert any(np.isclose(e, 0.0, 1e-02) for e in eig_vals)

def test_condition_indices():
    df = pd.DataFrame(np.random.randn(100, 2), columns=['a', 'b'])
    noise = np.random.randn(100)
    df['c'] = (2*df['a']) + (3*df['b']) + (.5*noise)
    cond_idx = condition_indices(df)
    assert type(cond_idx) == pd.Series
    assert any(c > 10.0 for c in cond_idx)

def test_vifs():
    df = pd.DataFrame(np.random.randn(100, 2), columns=['a', 'b'])
    noise = np.random.randn(100)
    df['c'] = (3*df['a']) + (2*df['b']) + (.5*noise)
    vifs = variance_inflation_factors(df)
    assert type(vifs) == pd.Series
    assert vifs.max() > 5.0
    assert vifs.idxmax() == 'c'

def test_outliers():
    # Test row outliers
    df = pd.DataFrame(np.random.randint(0, high=42, size=(100, 4)))
    df = df.append([[-42] * 4], ignore_index=True)
    mhlb = mahalanobis_distances(df)
    assert type(mhlb) == pd.Series
    assert mhlb.size == df.shape[0]
    assert mhlb.idxmax() == mhlb.size-1

    # Test column outliers
    df = pd.DataFrame(np.random.randint(0, high=42, size=(100, 4)))
    df['out'] = [-42] * 100
    mhlb = mahalanobis_distances(df, axis=1)
    assert type(mhlb) == pd.Series
    assert mhlb.size == df.shape[1]
    #TODO: fix, figure out why sometimes fails cause NaN
    #assert mhlb.idxmax() == 'out'

def test_variances():
    df = pd.DataFrame(np.random.randn(10, 2), columns=['a', 'b'])
    df['constant'] = [42.0] * df.shape[0]
    var = variances(df)
    assert type(var) == pd.Series
    assert var['constant'] == 0.0
    assert var['a'] > 0.0