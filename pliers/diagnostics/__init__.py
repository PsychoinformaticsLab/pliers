''' The `diagnostics` module contains functions for computing basic metrics
that may be of use in determining the quality of `Extractor` results. '''

from .base import Diagnostics
from .base import correlation_matrix
from .base import eigenvalues
from .base import condition_indices
from .base import variance_inflation_factors
from .base import mahalanobis_distances
from .base import variances


__all__ = [
    'correlation_matrix',
    'eigenvalues',
    'condition_indices',
    'variance_inflation_factors',
    'mahalanobis_distances',
    'variances',
    'Diagnostics'
]
