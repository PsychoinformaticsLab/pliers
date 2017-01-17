from pliers.extractors.base import Extractor, merge_results
from pliers.transformers import get_transformer
from itertools import chain
from pliers.utils import listify, flatten, isgenerator
from pliers import config
from six import string_types
from collections import OrderedDict


class Node(object):

    ''' A graph node/vertex. Represents a single transformer, optionally with
    references to children.
    Args:
        name (str): Name of the node
        transformer (Transformer): the Transformer instance at this node
    '''

    def __init__(self, transformer, name):
        self.name = name
        self.children = []
        if isinstance(transformer, string_types):
            transformer = get_transformer(transformer)
        self.transformer = transformer

    def add_child(self, node):
        ''' Append a child to the list of children. '''
        self.children.append(node)

    def is_leaf(self):
        return len(self.children)


class Graph(object):

    def __init__(self, nodes=None):

        self.nodes = OrderedDict()
        self.roots = []
        if nodes is not None:
            self.add_nodes(nodes)

    def add_nodes(self, nodes, parent=None):
        for n in nodes:
            node_args = self._parse_node_args(n)
            self.add_node(parent=parent, **node_args)

    def add_node(self, transformer, name=None, children=None, parent=None,
                 return_node=False):

        if name is None:
            name = id(transformer)

        node = Node(transformer, name)
        self.nodes[name] = node

        if parent is None:
            self.roots.append(node)
        else:
            parent = self.nodes[parent.name]
            parent.add_child(node)

        if children is not None:
            self.add_nodes(children, parent=node)

        if return_node:
            return node

    def run(self, stim, merge=True):
        results = list(chain(*[self.run_node(n, stim) for n in self.roots]))
        results = list(flatten(results))
        return merge_results(results) if merge else results

    def run_node(self, node, stim):

        if isinstance(node, string_types):
            node = self.nodes[node]

        result = node.transformer.transform(stim)
        if isinstance(node.transformer, Extractor):
            return listify(result)

        stim = result
        # If result is a generator, the first child will destroy the
        # iterable, so cache via list conversion
        if len(node.children) > 1 and isgenerator(stim):
            stim = list(stim)
        return list(chain(*[self.run_node(c, stim) for c in node.children]))

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
        else:
            kwargs['transformer'] = node

        return kwargs
