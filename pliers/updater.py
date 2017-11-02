""" Utility to check if results have changed in foreign APIs """
import glob
import datetime
import pandas as pd
from os.path import realpath, join, dirname, exists

from pliers.stimuli import load_stims
from pliers.graph import Graph

def check_updates(graph_spec, datastore, stimuli=None):
    """ Run graph_spec on set of stimuli, and store results in datastore csv.
    If file exists, compare results to previous in file,
    and return if any features have changed
    Args:
        graph_spec - A dictionary (or JSON file) graph spec.
        datastore - filepath of CSV file with results.
        stimuli - list of stimuli file paths to extract from.
                  if None, use test data.
    """
    # Check if file exists
    if not exists(datastore):
        prior_data = None
    else:
        prior_data = pd.read_csv(datastore)

    if stimuli is None:
        stimuli = glob.glob(join(
            dirname(
                realpath(__file__)), 'tests/data/image/CC0/*'))

    stimuli = load_stims(stimuli)
    graph = Graph(spec=graph_spec)
    results = graph.run(stimuli).drop(
        ['source_file', 'filename', 'history', 'class', 'onset', 'duration'], axis=1)
    results['time_extracted'] = datetime.datetime.now()

    # Flatten columns and make into one row
    results = results.pivot(columns='stim_name', index='time_extracted')
    results.columns = ['.'.join(col).strip() \
     if col[1] is not '' else col[0] for col in results.columns.values]
    results = results.reset_index()

    if prior_data is None:
        results.to_csv(datastore, index=False)
        return "File created"
    else:
        mismatches = []
        last = prior_data[
            prior_data.time_extracted == prior_data.time_extracted.max()].\
            iloc[0]

        for label, value in results.iteritems():
            if label != 'time_extracted':
                old_value = last.get(label)
                if (old_value is not None) and (old_value != value.values[0]):
                    assert 0
                    mismatches.append(label)

        return mismatches
