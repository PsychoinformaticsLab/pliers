""" Utility to check if results have changed in foreign APIs. """

import glob
import datetime
import pandas as pd
import numpy as np
from os.path import realpath, join, dirname, exists, expanduser
from pliers.stimuli import load_stims
from pliers.transformers import get_transformer
import hashlib
import pickle

def hash_data(data, blocksize=65536):
    """" Hashes list of data, strings or data """
    data = pickle.dumps(data)

    hasher = hashlib.sha1()
    hasher.update(data)

    return hasher.hexdigest()


def check_updates(transformers, datastore=None, stimuli=None):
    """ Run transformers through a battery of stimuli, and check if output has
    changed. Store results in csv file for comparison.

    Args:
        transformers (list): A list of tuples of transformer names and
            dictionary of parameters to instantiate with (or empty dict).
        datastore (str): Filepath of CSV file with results. Stored in home dir
            by default.
        stimuli (list): List of stimuli file paths to extract from. If None,
            use test data.
    """
    # Find datastore file
    datastore = datastore or expanduser('~/.pliers_updates')
    prior_data = pd.read_csv(datastore) if exists(datastore) else None

    # Load stimuli
    stimuli = stimuli or glob.glob(
        join(dirname(realpath(__file__)), '../tests/data/image/CC0/*'))
    stimuli = load_stims(stimuli)

    # Get transformers
    loaded_transformers = {get_transformer(name, **params): (name, params)
                           for name, params in transformers}

    # Transform stimuli
    results = pd.DataFrame({'time_extracted': [datetime.datetime.now()]})
    for trans in loaded_transformers.keys():
        for stim in stimuli:
            if trans._stim_matches_input_types(stim):
                res = trans.transform(stim)

                try: # Add iterable
                    res = [getattr(res, '_data', res.data) for r in res]
                except TypeError:
                    res = getattr(res, '_data', res.data)

                res = hash_data(res)

                results["{}.{}".format(trans.__hash__(), stim.name)] = [res]

    # Check for mismatches
    mismatches = []
    if prior_data is not None:
        last = prior_data[
            prior_data.time_extracted == prior_data.time_extracted.max()]. \
            iloc[0].drop('time_extracted')

        for label, value in results.iteritems():
            old = last.get(label)
            new = value.values[0]

            if old is not None:
                if isinstance(new, str):
                    if new != old:
                        mismatches.append(label)
                elif not np.isclose(old, new):
                    mismatches.append(label)

        results = prior_data.append(results)

    results.to_csv(datastore, index=False)

    # Get corresponding transformer name and parameters
    def get_trans(hash_tr):
        for obj, attr in loaded_transformers.items():
            if str(obj.__hash__()) == hash_tr:
                return attr

    delta_t = set([m.split('.')[0] for m in mismatches])
    delta_t = [get_trans(dt) for dt in delta_t]

    return {'transformers': delta_t, 'mismatches': mismatches}
