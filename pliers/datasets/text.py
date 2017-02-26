''' Functionality for loading and manipulating text datasets. '''

import zipfile
import json
import os
import tempfile
import io
import requests
import pandas as pd


def _load_datasets():
    path = os.path.abspath(__file__)
    path = os.path.join(os.path.dirname(path), 'dictionaries.json')
    dicts = json.load(io.open(path, encoding='utf-8'))
    return dicts

datasets = _load_datasets()


def _get_dictionary_path():
    # For now, stash everything under home directory.
    # TODO: Generalize this to support default system paths, env vars, etc.
    dir_path = os.path.expanduser(
        os.path.join('~', 'pliers_data', 'dictionaries'))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def _download_dictionary(url, format, rename):

    tmpdir = tempfile.mkdtemp()
    _file = os.path.join(tmpdir, os.path.basename(url))
    r = requests.get(url)
    with open(_file, 'wb') as f:
        f.write(r.content)

    if zipfile.is_zipfile(_file):
        with zipfile.ZipFile(_file) as zf:
            source = zf.namelist()[0]
            zf.extract(source, tmpdir)
            _file = os.path.join(tmpdir, source)

    if format == 'csv' or url.endswith('csv'):
        data = pd.read_csv(_file)
    elif format == 'tsv' or url.endswith('tsv'):
        data = pd.read_table(_file, sep='\t')
    elif format.startswith('xls') or os.path.splitext(url)[1].startswith('xls'):
        data = pd.read_excel(_file)

    if rename is not None:
        data = data.rename(columns=rename)
    return data


def fetch_dictionary(name, url=None, format=None, index=0, rename=None,
                     save=True):
    ''' Retrieve a dictionary of text norms from the web or local storage.
    Args:
        name (str): The name of the dictionary. If no url is passed, this must
            match either one of the keys in the predefined dictionary file (see
            dictionaries.json), or the name assigned to a previous dictionary
            retrieved from a specific URL.
        url (str): The URL of dictionary file to retrieve. Optional if name
            matches an existing dictionary.
        format (str): One of 'csv', 'tsv', 'xls', or None. Used to read data
            appropriately. Note that most forms of compression will be detected
            and handled automatically, so the format string refers only to the
            format of the decompressed file. When format is None, the format
            will be inferred from the filename.
        index (str, int): The name or numeric index of the column to used as
            the dictionary index. Passed directly to pd.ix.
        rename (dict): An optional dictionary passed to pd.rename(); can be
            used to rename columns in the loaded dictionary. Note that the
            locally-saved dictionary will retain the renamed columns.
        save (bool): Whether or not to save the dictionary locally the first
            time it is retrieved.
    Returns: A pandas DataFrame indexed by strings (typically words).

    '''
    file_path = os.path.join(_get_dictionary_path(), name + '.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        index = datasets[name].get('index', df.columns[index])
        return df.set_index(index)

    elif name in datasets:
        url = datasets[name]['url']
        format = datasets[name].get('format', format)
        index = datasets[name].get('index', index)
        rename = datasets.get('rename', rename)

    if url is None:
        raise ValueError("Dataset '%s' not found in local storage or presets, "
                         "and no download URL provided." % name)
    data = _download_dictionary(url, format=format, rename=rename)

    if isinstance(index, int):
        index = data.columns[index]
    data = data.set_index(index)

    if save:
        file_path = os.path.join(_get_dictionary_path(), name + '.csv')
        data.to_csv(file_path, encoding='utf-8')
    return data
