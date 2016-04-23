import urllib
import pandas as pd
import json
import os
import tempfile
import zipfile
import shutil


def _load_datasets():
    path = os.path.abspath(__file__)
    path = os.path.join(os.path.dirname(path), 'dictionaries.json')
    dicts = json.load(open(path))
    return dicts

datasets = _load_datasets()


def _get_dictionary_path():
    # For now, stash everything under home directory.
    # TODO: Generalize this to support default system paths, env vars, etc.
    dir_path = os.path.expanduser(os.path.join('~', 'featurex_data', 'dictionaries'))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def _download_dictionary(url, format=None, rename=None, save=None):

    tmpdir = tempfile.mkdtemp()
    _file = os.path.join(tmpdir, os.path.basename(url))
    urllib.urlretrieve(url, _file)

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
    if save is not None:
        file_path = os.path.join(_get_dictionary_path(), save + '.csv')
        data.to_csv(file_path, index=False, encoding='utf-8')
    return data


def fetch_dictionary(name, url=None, format='csv', rename=None):
    file_path = os.path.join(_get_dictionary_path(), name + '.csv')
    if os.path.exists(file_path):
        return pd.read_csv(file_path)

    elif name in datasets:
        url = datasets[name]['url']
        format = datasets[name].get('format', format)
        rename = datasets.get('rename', rename)

    if url is None:
        raise ValueError("Dataset '%s' not found in local storage or presets, "
                         "and no download URL provided." % name)
    return _download_dictionary(url, format=format, rename=rename, save=name)


if __name__ == '__main__':
    print(fetch_dictionary('aoa'))