from pliers.datasets.text import _load_datasets
import requests


def test_dicts_exist_at_url_and_initialize():
    """
    Check that all text dictionaries download successfully.
    """
    datasets = _load_datasets()
    for name, dataset in datasets.items():
        r = requests.head(dataset['url'])
        assert r.status_code in (200, 301)
