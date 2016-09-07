from featurex.datasets.text import _load_datasets, fetch_dictionary
from unittest import TestCase
from pandas import DataFrame
import urllib2


class TestDatasets(TestCase):
    
    def test_dicts(self):
        """
        Check that all text dictionaries download successfully.
        """
        datasets = _load_datasets()
        for dataset in datasets.keys():
            try:
                data = fetch_dictionary(dataset, save=False)
            except:
                print("Dataset failed: {0}".format(dataset))
                data = None
                
                # Determine cause of error.
                try:
                    urllib2.urlopen(datasets[dataset]["url"])
                except urllib2.HTTPError, e:
                    print("HTTP Error: {0}".format(e.code))
                except urllib2.URLError, e:
                    print("URL Error: {0}".format(e.args))
            self.assertIsInstance(data, DataFrame)
