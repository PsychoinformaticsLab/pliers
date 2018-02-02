from .config import set_option, get_option, set_options
from .graph import Graph
from .version import __version__

__all__ = [
    'config'
    'set_option',
    'set_options',
    'get_option',
    'graph',
    'transformers',
    'utils',
    'converters',
    'datasets',
    'diagnostics',
    'external',
    'extractors',
    'filters',
    'stimuli',
    'support',
    'Graph'
]

from .support.due import due, Url

# TODO: replace with Doi whenever available (Zenodo?)
due.cite(
    Url("https://github.com/tyarkoni/pliers"),
    description="A Python package for automated extraction of features from multimodal stimuli",
    tags=['reference-implementation'],
    path='pliers'
)