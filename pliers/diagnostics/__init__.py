''' The `diagnostics` module contains functions for computing basic metrics
that may be of use in determining the quality of `Extractor` results. '''

from .diagnostics import Diagnostics
from .diagnostics import correlation_matrix
from .diagnostics import eigenvalues
from .diagnostics import condition_indices
from .diagnostics import variance_inflation_factors
from .diagnostics import mahalanobis_distances
from .diagnostics import variances


__all__ = [
    'correlation_matrix',
    'eigenvalues',
    'condition_indices',
    'variance_inflation_factors',
    'mahalanobis_distances',
    'variances',
    'Diagnostics'
]
