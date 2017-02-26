from pliers.extractors.base import Extractor, merge_results
from pliers.transformers import get_transformer
from itertools import chain
from pliers.utils import listify, flatten, isgenerator
from six import string_types
from collections import OrderedDict

try:
    import pygraphviz as pgv
except:
    pgv = None


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

    def draw(self, filename):
        ''' Render a plot of the graph via pygraphviz.
        Args:
            filename (str): Path to save the generated image to.
        '''
        if pgv is None:
            raise ImportError("pygraphviz is required in order to plot graphs,"
                              " but could not be successfully imported. Please"
                              " make sure it is installed.")
        if not hasattr(self, '_results'):
            raise RuntimeError("Graph cannot be drawn before it is executed. "
                             "Try calling run() first.")

        g = pgv.AGraph(directed=True)
        node_list = {}

        for elem in self._results:
            if not hasattr(elem, 'history'):
                continue
            log = elem.history

            has_parent = True

            while has_parent:

                # Add nodes
                source_from = log.parent[6] if log.parent else ''
                s_node = hash((source_from, log[2]))
                if s_node not in node_list:
                    g.add_node(s_node, label=log[2], shape='ellipse')

                t_node = hash((log[6], log[7]))
                if t_node not in node_list:
                    g.add_node(t_node, label=log[6], shape='box')

                r_node = hash((log[6], log[5]))
                if r_node not in node_list:
                    g.add_node(r_node, label=log[5], shape='ellipse')

                # Add edges
                g.add_edge(s_node, t_node)
                g.add_edge(t_node, r_node)
                has_parent = log.parent
                log = log.parent

        g.draw(filename, prog='dot')

    def run(self, stim, merge=True):
        results = list(chain(*[self.run_node(n, stim) for n in self.roots]))
        results = list(flatten(results))
        self._results = results  # For use in plotting
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
