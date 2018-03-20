from pliers.utils.updater import check_updates
from tempfile import NamedTemporaryFile
import pandas as pd

def test_updater():
    datastore_file = NamedTemporaryFile().name
    transformers = [('BrightnessExtractor', {}), ('TesseractConverter', {})]
    # Run updater once
    results = check_updates(transformers, datastore=datastore_file)
    assert results == {'transformers': [], 'mismatches': []}

    # Run again with same values
    results = check_updates(transformers, datastore=datastore_file)
    assert results == {'transformers': [], 'mismatches': []}

    # Change value in datastore
    ds = pd.read_csv(datastore_file)
    ds.iloc[1, 1:] = 1
    ds.to_csv(datastore_file, index=False)

    results = check_updates(transformers, datastore=datastore_file)
    for j in results['transformers']:
        assert j in transformers
    assert len(results['mismatches']) == 198
