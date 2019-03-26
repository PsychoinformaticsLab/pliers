''' The config module contains package-level settings and tools for
manipulating them. '''

import json
from os.path import join, expanduser, exists
import os
from six import string_types

__all__ = ['set_option', 'set_options', 'get_option']

_config_name = 'pliers_config.json'

_default_converters = {
    'AudioStim->TextStim':
        ('GoogleSpeechAPIConverter', 'IBMSpeechAPIConverter',
         'WitTranscriptionConverter'),
    'ImageStim->TextStim':
        ('GoogleVisionAPITextConverter', 'TesseractConverter')
}

_default_settings = {
    'cache_transformers': True,
    'default_converters': _default_converters,
    'drop_bad_extractor_results': True,
    'log_transformations': True,
    'n_jobs': None,
    'parallelize': False,
    'progress_bar': True,
    'use_generators': False,
    'allow_large_jobs': True,
    'long_job': 60,  # in seconds
    'large_job': 100,
    'api_key_validation': False
}


def set_option(key, value):
    if key not in _settings:
        raise ValueError("Invalid pliers setting: '%s'" % key)
    _settings[key] = value


def set_options(**kwargs):
    for k, v in kwargs.items():
        set_option(k, v)


def get_option(key):
    if key not in _settings:
        raise ValueError("Invalid pliers setting: '%s'" % key)
    return _settings[key]


def from_file(filenames, error_on_missing=True):
    if isinstance(filenames, string_types):
        filenames = [filenames]
    for f in filenames:
        if exists(f):
            settings = json.load(open(f))
            _settings.update(settings)
        elif error_on_missing:
            raise ValueError("Config file '%s' does not exist." % f)


def reset_options(update_from_file=False):
    ''' Reset all options to the package defaults.
    Args:
        from_file (bool): If True, re-applies any config files found in
            standard locations.
    '''
    global _settings
    _settings = _default_settings.copy()
    if update_from_file:
        _update_from_standard_locations()


def _update_from_standard_locations():
    ''' Check standard locations for config files and update settings if found.
    Order is user's home dir, environment variable ($PLIERS_CONFIG), and then
    current directory--with later files taking precedence over earlier ones.
    '''
    locs = [
        join(expanduser('~'), _config_name),
        join('.', _config_name)
    ]
    if 'PLIERS_CONFIG' in os.environ:
        locs.insert(1, os.environ['PLIERS_CONFIG'])

    from_file(locs, False)


_settings = {}
reset_options(True)
