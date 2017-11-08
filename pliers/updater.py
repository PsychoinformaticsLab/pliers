""" Utility to check if results have changed in foreign APIs """
import glob
import datetime
import warnings
import pandas as pd
import numpy as np
import json
from os.path import realpath, join, dirname, exists

from pliers.stimuli import load_stims
from pliers.graph import Graph
from pliers.extractors import merge_results

from copy import deepcopy

def flatten_mixed_list(mixed):
    """ Flattens a list of objects and lists """
    flat_list = []
    for item in mixed:
        flat_list += item if isinstance(item, list) else [item]
    return flat_list

def filter_nodes_stimuli(nodes, stimulus):
    """ Recursively filter nodes against stimulus type.
    If node does not match input type, remove and replace with children.
    Args:
        nodes - A list of nodes
        stimulus - A stimulus to match against
    """
    filtered_nodes = []
    if nodes:
        for node in nodes:
            children = filter_nodes_stimuli(node.children, stimulus)
            if not node.transformer._stim_matches_input_types(stimulus):
                warnings.warn("Node {} incompatible, removed.".format(node.transformer.name))
                node = deepcopy(children)
            else:
                node.children = deepcopy(children)
            filtered_nodes.append(node)

    return flatten_mixed_list(filtered_nodes) or None

def filter_nodes_extractor(nodes, extractors):
    """ Recursively search nodes to find extractors.
    Keep nodes with matching children.
    Args:
        nodes - A list of nodes in dictonary (JSON) format
        extractors - List of extractors to match against
    """
    filtered_nodes = []
    for node in nodes:
        if 'children' in node:
            node['children'] = filter_nodes_extractor(node['children'], extractors)
        if node.get('children') or node['transformer'] in extractors:
            filtered_nodes.append(node)

    return filtered_nodes

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
        stim_graph = Graph(
            nodes=filter_nodes_stimuli(graph.roots, stim))
        results += stim_graph.run(stim, merge=False)

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

    # Filter original graph on extractors
    if extractors:
        graph_json = json.load(open(graph_spec, 'r'))
        graph_json['roots'] = filter_nodes_extractor(
            graph_json['roots'], extractors)
    else:
        graph_json = None

    return {'difference_graph': graph_json,
            'changed_extractors' : extractors or None,
            'mismatches' : mismatches or None}
