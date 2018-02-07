from .config import set_option, get_option, set_options
from .graph import Graph
from .version import __version__
from .support.due import due, Url, BibTeX


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

# TODO: replace with Doi whenever available (Zenodo?)
due.cite(
    Url("https://github.com/tyarkoni/pliers"),
    description="A Python package for automated extraction of features from "
                "multimodal stimuli",
    tags=['reference-implementation'],
    path='pliers'
)

due.cite(BibTeX("""
    @inproceedings{McNamara:2017:DCF:3097983.3098075,
     author = {McNamara, Quinten and De La Vega, Alejandro and Yarkoni, Tal},
     title = {Developing a Comprehensive Framework for Multimodal Feature Extraction},
     booktitle = {Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
     series = {KDD '17},
     year = {2017},
     isbn = {978-1-4503-4887-4},
     location = {Halifax, NS, Canada},
     pages = {1567--1574},
     numpages = {8},
     url = {http://doi.acm.org/10.1145/3097983.3098075},
     doi = {10.1145/3097983.3098075},
     acmid = {3098075},
     publisher = {ACM},
     address = {New York, NY, USA},
     keywords = {feature extraction, multimodal retrieval, python, standardization, wrappers},
    }"""),
    description="Developing a Comprehensive Framework for Multimodal Feature Extraction", path="pliers")
