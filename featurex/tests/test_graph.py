import pytest
from featurex.graph import Graph, Node
from featurex.converters.image import TesseractConverter
from featurex.converters.video import FrameSamplingConverter, VideoToAudioConverter
from featurex.converters.api import WitTranscriptionConverter
from featurex.extractors.image import BrightnessExtractor, VibranceExtractor
from featurex.extractors.text import LengthExtractor
from featurex.stimuli.image import ImageStim
from featurex.stimuli.video import VideoStim
from .utils import get_test_data_path, DummyExtractor
from os.path import join
import numpy as np
from numpy.testing import assert_almost_equal


def test_node_init():
    n = Node('my_node', BrightnessExtractor())
    assert isinstance(n.transformer, BrightnessExtractor)
    assert n.name == 'my_node'
    n = Node('my_node', 'brightnessextractor')    
    assert isinstance(n.transformer, BrightnessExtractor)


def test_node_arg_parsing():
    n1, n2 = 'MyLovelyExtractor', ['MyLovelyExtractor']
    args1 = Graph._parse_node_args(n1)
    args2 = Graph._parse_node_args(n2)
    assert args1 == args2 == {'transformer': 'MyLovelyExtractor'}

    node = ('saliencyextractor', 'saliency')
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
    nodes = [(BrightnessExtractor(), 'brightness')]
    graph = Graph(nodes)
    result = graph.extract([stim])
    brightness = result[('BrightnessExtractor', 'brightness')].values[0]
    assert_almost_equal(brightness, 0.556134, 5)


def test_add_children():
    graph = Graph()
    de1, de2, de3 = DummyExtractor(), DummyExtractor(), DummyExtractor()
    graph.add_children([de1, de2, de3])
    assert len(graph.children) == 3
    assert all([isinstance(c, Node) for c in graph.children])


def test_add_nested_children():
    graph = Graph()
    de1, de2, de3 = DummyExtractor(), DummyExtractor(), DummyExtractor()
    graph.add_children([de1, (de2, 'child', [(de3, "child's child")])])
    assert len(graph.children) == 2
    assert isinstance(graph.children[1].children[0], Node)
    assert graph.children[1].children[0].name == "child's child"


def test_small_pipeline():
    pytest.importorskip('pytesseract')
    filename = join(get_test_data_path(), 'image', 'button.jpg')
    stim = ImageStim(filename)
    nodes = [(TesseractConverter(), 'tesseract', 
                [(LengthExtractor(), 'length')])]
    graph = Graph(nodes)
    result = graph.extract([stim])
    assert (0, 'button.jpg_Exit') in result.index
    assert ('LengthExtractor', 'text_length') in result.columns
    assert result[('LengthExtractor', 'text_length')].values[0] == 4


@pytest.mark.skipif("'WIT_AI_API_KEY' not in os.environ")
def test_big_pipeline():
    filename = join(get_test_data_path(), 'video', 'obama_speech.mp4')
    video = VideoStim(filename)
    visual_nodes = [(FrameSamplingConverter(every=15), 'framesampling', 
                    [(TesseractConverter(), 'visual_text', 
                    [(LengthExtractor(), 'text_length')]), 
                    (VibranceExtractor(), 'visual_vibrance')])]
    audio_nodes = [(VideoToAudioConverter(), 'audio', 
                    [(WitTranscriptionConverter(), 'audio_text', 
                    [(LengthExtractor(), 'text_length')])])]

    graph = Graph()
    graph.add_children(visual_nodes)
    graph.add_children(audio_nodes)
    result = graph.extract(video)
    assert ('LengthExtractor', 'text_length') in result.columns
    assert ('VibranceExtractor', 'vibrance') in result.columns
    #TODO: make this have better checks
