import pytest
from featurex.graph import Graph, Node
from featurex.extractors.image import BrightnessExtractor, SaliencyExtractor
from featurex.stimuli.image import ImageStim
from .utils import get_test_data_path
from os.path import join


def test_node_init():
    n = Node('my_node', SaliencyExtractor())
    assert isinstance(n.transformer, SaliencyExtractor)
    assert n.name == 'my_node'
    n = Node('my_node', 'saliencyextractor')    
    assert isinstance(n.transformer, SaliencyExtractor)


def test_node_arg_parsing():
    n1, n2 = 'MyLovelyExtractor', ['MyLovelyExtractor']
    args1 = Graph._parse_node_args(n1)
    args2 = Graph._parse_node_args(n2)
    assert args1 == args2 == {'transformer': 'MyLovelyExtractor'}

    node = (SaliencyExtractor(), 'saliency')
    args = Graph._parse_node_args(node)
    assert set(args.keys()) == {'transformer', 'name'}

    node = ('saliencyextractor', 'my_name', [('child1'), ('child2')])
    args = Graph._parse_node_args(node)
    assert set(args.keys()) == {'transformer', 'name', 'children'}
    assert len(args['children']) == 2

    node = { 'transformer': '...', 'name': '...'}
    args = Graph._parse_node_args(node)
    assert args == node


def test_graph_smoke_test():
    
    filename = join(get_test_data_path(), 'image', 'obama.jpg')
    stim = ImageStim(filename)
    nodes = [
        (BrightnessExtractor(), 'brightness'),
        'saliencyextractor'
    ]
    graph = Graph(nodes)
    result = graph.extract([stim])
    assert 'avg_brightness' in result[0].data
    assert len(set(result[1].data.keys()) & {'max_saliency', 'max_x'}) == 2
