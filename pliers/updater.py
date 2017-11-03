""" Utility to check if results have changed in foreign APIs """
import glob
import datetime
import warnings
import pandas as pd
import numpy as np
from os.path import realpath, join, dirname, exists

from pliers.stimuli import load_stims
from pliers.graph import Graph
from pliers.extractors import merge_results

from copy import deepcopy

def flatten_mixed_list(mixed):
    flat_list = []
    for subitem in mixed:
        if isinstance(subitem, list):
            for item in subitem:
                flat_list.append(item)
        else:
            flat_list.append(subitem)
    return flat_list

def filter_incompatible_nodes(nodes, stimulus):
    """ Recursively filter nodes against stimulus type """
    filtered_nodes = []
    if nodes:
        for node in nodes:
            children = filter_incompatible_nodes(node.children, stimulus)
            if not node.transformer._stim_matches_input_types(stimulus):
                warnings.warn("Node {} incompatible, removed.".format(node.transformer.name))
                node = deepcopy(children)
            else:
                node.children = deepcopy(children)
            filtered_nodes.append(node)

    return flatten_mixed_list(filtered_nodes) or None



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

    # Load stimuli and extract new features
    stimuli = load_stims(stimuli)
    graph = Graph(spec=graph_spec)

    results = []
    for stim in stimuli:
        stim_graph = Graph(nodes=filter_incompatible_nodes(graph.roots, stim))
        results.append(stim_graph.run(stim, merge=False))

    ## Merge results
    results = merge_results(results)
    # assert 0
    results = results.drop(
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
                old_value = last.get(label)
                new_value = value.values[0]

                # This this value was recorded last time
                if old_value is not None and not np.isclose(old_value, new_value):
                    mismatches.append(label)

        new_data = prior_data.append(results)
        new_data.to_csv(datastore, index=False)

    extractors = set([m.split('.')[0] for m in mismatches])

    return {'changed_extractors' : extractors, 'mismatches' : mismatches}
