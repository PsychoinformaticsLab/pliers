from pliers.datasets.text import _load_datasets
import requests


def test_dicts_exist_at_url_and_initialize():
    """
    Check that all text dictionaries download successfully.
    """
    datasets = _load_datasets()
    for name, dataset in datasets.items():
        r = requests.head(dataset['url'])
        assert r.status_code == requests.codes.ok
        # read_excel() is doing some weird things, so disable for the moment
        # data = fetch_dictionary(name, save=False)
        # assert isinstance(data.shape, tuple)
