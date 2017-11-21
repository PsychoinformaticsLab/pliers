""" Utility to check if results have changed in foreign APIs """
import glob
import datetime
import warnings
import pandas as pd
import numpy as np
from os.path import realpath, join, dirname, exists, expanduser

from pliers.stimuli import load_stims
from pliers.extractors import merge_results

def check_updates(extractors, datastore=None, stimuli=None):
    """ Run graph_spec on set of stimuli, and store results in datastore csv.
    If file exists, compare results to previous in file,
    and return if any features have changed
    Args:
        extractors - A list of extractor objets.
        datastore - filepath of CSV file with results. Stored in home dir
                    by default
        stimuli - list of stimuli file paths to extract from.
                  if None, use test data.
    """
    # Check if file exists
    if datastore is None:
        datastore = expanduser('~/.pliers_updates')
    if not exists(datastore):
        prior_data = None
    else:
        prior_data = pd.read_csv(datastore)

    if stimuli is None:
        stimuli = glob.glob(join(
            dirname(
                realpath(__file__)), '../tests/data/image/CC0/*'))

    # Load stimuli and extract new features
    stimuli = load_stims(stimuli)

    results = []
    for stim in stimuli:
        for ext in extractors:
            results.append(ext.transform(stim))

    ## Merge results
    results = merge_results(results).drop(
        ['source_file', 'filename', 'history', 'class', 'onset', 'duration'],
        axis=1, level=0)
    results['time_extracted'] = datetime.datetime.now()

    # Flatten columns and make into one row
    results = results.pivot(columns='stim_name', index='time_extracted')
    results.columns = ['.'.join(col).strip() \
     if col[1] is not '' else col[0] for col in results.columns.values]
    results = results.reset_index()

    # If new, record data
    mismatches = []
    if prior_data is None:
        results.to_csv(datastore, index=False)
        warnings.warn("Datastore not found. Initiated new.")
    else: # Otherwise look for mistmatches
        last = prior_data[
            prior_data.time_extracted == prior_data.time_extracted.max()].\
            iloc[0]

        for label, value in results.iteritems():
            if label != 'time_extracted':
                old = last.get(label)
                new = value.values[0]

                if old is not None and not np.isclose(old, new):
                    mismatches.append(label)

        new_data = prior_data.append(results)
        new_data.to_csv(datastore, index=False)

    extractors = set([m.split('.')[0] for m in mismatches])

    return {'changed_extractors' : extractors or None,
            'mismatches' : mismatches or None}
