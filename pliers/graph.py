''' The `graph` module contains tools for constructing and executing graphs
of pliers Transformers. '''

from pliers.extractors.base import merge_results
from pliers.stimuli import __all__ as stim_list
from pliers.transformers import get_transformer
from pliers.utils import (listify, flatten, isgenerator, attempt_to_import,
                          verify_dependencies)
from itertools import chain
from six import string_types
from collections import OrderedDict

import json

pgv = attempt_to_import('pygraphviz', 'pgv')
stim_list.insert(0, 'ExtractorResult')


class Node(object):

    ''' A graph node/vertex. Represents a single transformer, optionally with
    references to children.

    Args:
        name (str): Name of the node
        transformer (Transformer): the Transformer instance at this node
        parameters (kwargs): parameters for initializing the Transformer
    '''

    def __init__(self, transformer, name=None, **parameters):
        self.name = name
        self.children = []
        if isinstance(transformer, string_types):
            transformer = get_transformer(transformer, **parameters)
        self.transformer = transformer
        self.parameters = parameters
        if name is not None:
            self.transformer.name = name
        self.id = id(transformer)

    def add_child(self, node):
        ''' Append a child to the list of children. '''
        self.children.append(node)

    def is_leaf(self):
        return len(self.children) == 0

    def to_json(self):
        spec = {'transformer': self.transformer.__class__.__name__}
        if self.name:
            spec['name'] = self.name
        if self.children:
            children = []
            for c in self.children:
                children.append(c.to_json())
            spec['children'] = children
        if self.parameters:
            spec['parameters'] = self.parameters
        return spec


class Graph(object):
    ''' Graph-like structure that represents an entire pliers workflow.

    Args:
        nodes (list, dict): Optional nodes to add to the Graph at construction.
            If a dict, must have a 'roots' key. If a list, each element must be
            in one of the forms accepted by add_nodes().
        spec (str): An optional path to a .json file containing the graph
            specification.
    '''

    def __init__(self, nodes=None, spec=None):

        self.nodes = OrderedDict()
        self.roots = []
        if nodes is not None:
            if isinstance(nodes, dict):
                nodes = nodes['roots']
            self.add_nodes(nodes)
        elif spec is not None:
            with open(spec) as spec_file:
                self.add_nodes(json.load(spec_file)['roots'])

    @staticmethod
    def _parse_node_args(node):
        if isinstance(node, dict):
            return node

        kwargs = {}

        if isinstance(node, (list, tuple)):
            kwargs['transformer'] = node[0]
            if len(node) > 1:
                kwargs['children'] = node[1]
            if len(node) > 2:
                kwargs['name'] = node[2]
        elif isinstance(node, Node):
            kwargs['transformer'] = node.transformer
            kwargs['children'] = node.children
            kwargs['name'] = node.name
        else:
            kwargs['transformer'] = node

        return kwargs

    def add_nodes(self, nodes, parent=None, mode='horizontal'):
        ''' Adds one or more nodes to the current graph.

        Args:
            nodes (list): A list of nodes to add. Each element must be one of
                the following:

                * A dict containing keyword args to pass onto to the Node init.
                * An iterable containing 1 - 3 elements. The first element is
                  mandatory, and specifies the Transformer at that node. The
                  second element (optional) is an iterable of child nodes
                  (specified in the same format). The third element
                  (optional) is a string giving the (unique) name of the
                  node.
                * A Node instance.
                * A Transformer instance.

            parent (Node): Optional parent node (i.e., the node containing the
                pliers Transformer from which the to-be-created nodes receive
                their inputs).
            mode (str): Indicates the direction with which to add the new nodes
                * horizontal: the nodes should each be added as a child of the
                  'parent' argument (or a Graph root by default).
                * vertical: the nodes should each be added in sequence with
                  the first node being the child of the 'parnet' argument
                  (a Graph root by default) and each subsequent node being
                  the child of the previous node in the list.
        '''
        for n in nodes:
            node_args = self._parse_node_args(n)
            if mode == 'horizontal':
                self.add_node(parent=parent, **node_args)
            elif mode == 'vertical':
                parent = self.add_node(parent=parent, return_node=True,
                                       **node_args)
            else:
                raise ValueError("Invalid mode for adding nodes to a graph:"
                                 "%s" % mode)

    def add_chain(self, nodes, parent=None):
        ''' An alias for add_nodes with the mode preset to 'vertical'. '''
        self.add_nodes(nodes, parent, 'vertical')

    def add_children(self, nodes, parent=None):
        ''' An alias for add_nodes with the mode preset to 'horizontal'. '''
        self.add_nodes(nodes, parent, 'horizontal')

    def add_node(self, transformer, name=None, children=None, parent=None,
                 parameters={}, return_node=False):
        ''' Adds a node to the current graph.

        Args:
            transformer (str, Transformer): The pliers Transformer to use at
                the to-be-added node. Either a case-insensitive string giving
                the name of a Transformer class, or an initialized Transformer
                instance.
            name (str): Optional name to give this Node.
            children (list): Optional list of child nodes (i.e., nodes to pass
                the to-be-added node's Transformer output to).
            parent (Node): Optional node from which the to-be-added Node
                receives its input.
            parameters (dict): Optional keyword arguments to pass onto the
                Transformer initialized at this Node if a string is passed to
                the 'transformer' argument. Ignored if an already-initialized
                Transformer is passed.
            return_node (bool): If True, returns the initialized Node instance.

        Returns:
            The initialized Node instance if return_node is True,
                None otherwise.
        '''

        node = Node(transformer, name, **parameters)
        self.nodes[node.id] = node

        if parent is None:
            self.roots.append(node)
        else:
            parent = self.nodes[parent.id]
            parent.add_child(node)

        if children is not None:
            self.add_nodes(children, parent=node)

        if return_node:
            return node

    def run(self, stim, merge=True, **merge_kwargs):
        ''' Executes the graph by calling all Transformers in sequence.

        Args:
            stim (str, Stim, list): One or more valid inputs to any
                Transformer's 'transform' call.
            merge (bool): If True, all results are merged into a single pandas
                DataFrame before being returned. If False, a list of
                ExtractorResult objects is returned (one per Extractor/Stim
                combination).
            merge_kwargs: Optional keyword arguments to pass onto the
                merge_results() call.
        '''
        results = list(chain(*[self.run_node(n, stim) for n in self.roots]))
        results = list(flatten(results))
        self._results = results  # For use in plotting
        return merge_results(results, **merge_kwargs) if merge else results

    transform = run

    def run_node(self, node, stim):
        ''' Executes the Transformer at a specific node.

        Args:
            node (str, Node): If a string, the name of the Node in the current
                Graph. Otherwise the Node instance to execute.
            stim (str, stim, list): Any valid input to the Transformer stored
                at the target node.
        '''
        if isinstance(node, string_types):
            node = self.nodes[node]

        result = node.transformer.transform(stim)
        if node.is_leaf():
            return listify(result)

        stim = result
        # If result is a generator, the first child will destroy the
        # iterable, so cache via list conversion
        if len(node.children) > 1 and isgenerator(stim):
            stim = list(stim)
        return list(chain(*[self.run_node(c, stim) for c in node.children]))

    def draw(self, filename, color=True):
        ''' Render a plot of the graph via pygraphviz.

        Args:
            filename (str): Path to save the generated image to.
            color (bool): If True, will color graph nodes based on their type,
                otherwise will draw a black-and-white graph.
        '''
        verify_dependencies(['pgv'])
        if not hasattr(self, '_results'):
            raise RuntimeError("Graph cannot be drawn before it is executed. "
                               "Try calling run() first.")

        g = pgv.AGraph(directed=True)
        g.node_attr['colorscheme'] = 'set312'

        for elem in self._results:
            if not hasattr(elem, 'history'):
                continue
            log = elem.history

            while log:
                # Configure nodes
                source_from = log.parent[6] if log.parent else ''
                s_node = hash((source_from, log[2]))
                s_color = stim_list.index(log[2])
                s_color = s_color % 12 + 1

                t_node = hash((log[6], log[7]))
                t_style = 'filled,' if color else ''
                t_style += 'dotted' if log.implicit else ''
                if log[6].endswith('Extractor'):
                    t_color = '#0082c8'
                elif log[6].endswith('Filter'):
                    t_color = '#e6194b'
                else:
                    t_color = '#3cb44b'

                r_node = hash((log[6], log[5]))
                r_color = stim_list.index(log[5])
                r_color = r_color % 12 + 1

                # Add nodes
                if color:
                    g.add_node(s_node, label=log[2], shape='ellipse',
                               style='filled', fillcolor=s_color)
                    g.add_node(t_node, label=log[6], shape='box',
                               style=t_style, fillcolor=t_color)
                    g.add_node(r_node, label=log[5], shape='ellipse',
                               style='filled', fillcolor=r_color)
                else:
                    g.add_node(s_node, label=log[2], shape='ellipse')
                    g.add_node(t_node, label=log[6], shape='box',
                               style=t_style)
                    g.add_node(r_node, label=log[5], shape='ellipse')

                # Add edges
                g.add_edge(s_node, t_node, style=t_style)
                g.add_edge(t_node, r_node, style=t_style)
                log = log.parent

        g.draw(filename, prog='dot')

    def to_json(self):
        ''' Returns the JSON representation of this graph. '''
        roots = []
        for r in self.roots:
            roots.append(r.to_json())
        return {'roots': roots}

    def save(self, filename):
        ''' Writes the JSON representation of this graph to the provided
        filename, such that the graph can be easily reconstructed using
        Graph(spec=filename).

        Args:
            filename (str): Path at which to write out the json file.
        '''
        with open(filename, 'w') as outfile:
            json.dump(self.to_json(), outfile)
