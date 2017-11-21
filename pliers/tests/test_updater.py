from pliers.utils.updater import check_updates
from .utils import DummyExactExtractor
from tempfile import NamedTemporaryFile

def test_updater():
    datastore_file = NamedTemporaryFile().name
    # Run updater once
    results = check_updates([DummyExactExtractor(5)], datastore=datastore_file)
    assert results == {'changed_extractors': None, 'mismatches': None}

    # Run again with same values
    results = check_updates([DummyExactExtractor(5)], datastore=datastore_file)
    assert results == {'changed_extractors': None, 'mismatches': None}

    # Change value
    results = check_updates([DummyExactExtractor(6)], datastore=datastore_file)
    assert results['changed_extractors'] == {'DummyExactExtractor'}
    assert len(results['mismatches']) > 10
