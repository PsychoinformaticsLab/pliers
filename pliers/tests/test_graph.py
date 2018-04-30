import pytest
from pliers.graph import Graph, Node
from pliers.converters import (TesseractConverter,
                               VideoToAudioConverter,
                               WitTranscriptionConverter)
from pliers.filters import FrameSamplingFilter
from pliers.extractors import (BrightnessExtractor, VibranceExtractor,
                               LengthExtractor, merge_results)
from pliers.stimuli import (ImageStim, TextStim, VideoStim)
from .utils import get_test_data_path, DummyExtractor
from os.path import join, exists
from numpy.testing import assert_almost_equal
import pandas as pd
import tempfile
import os


def test_node_init():
    n = Node(BrightnessExtractor(), 'my_node')
    assert isinstance(n.transformer, BrightnessExtractor)
    assert n.name == 'my_node'
    n = Node('brightnessextractor', 'my_node')
    assert isinstance(n.transformer, BrightnessExtractor)


def test_node_arg_parsing():
    n1, n2 = 'MyLovelyExtractor', ['MyLovelyExtractor']
    args1 = Graph._parse_node_args(n1)
    args2 = Graph._parse_node_args(n2)
    assert args1 == args2 == {'transformer': 'MyLovelyExtractor'}

    node = ('saliencyextractor', [])
    args = Graph._parse_node_args(node)
    assert set(args.keys()) == {'transformer', 'children'}

    node = ('saliencyextractor', [('child1'), ('child2')], 'my_name')
    args = Graph._parse_node_args(node)
    assert set(args.keys()) == {'transformer', 'name', 'children'}
    assert len(args['children']) == 2

    node = {'transformer': '...', 'name': '...'}
    args = Graph._parse_node_args(node)
    assert args == node


def test_graph_smoke_test():
    filename = join(get_test_data_path(), 'image', 'obama.jpg')
    stim = ImageStim(filename)
    nodes = [(BrightnessExtractor(), [], 'brightness_node')]
    graph = Graph(nodes)
    result = graph.run(stim, format='wide', extractor_names='multi')
    brightness = result[('brightness_node', 'brightness')].values[0]
    assert_almost_equal(brightness, 0.556134, 5)


def test_add_children():
    graph = Graph()
    de1, de2, de3 = DummyExtractor(), DummyExtractor(), DummyExtractor()
    graph.add_nodes([de1, de2, de3])
    assert len(graph.roots) == 3
    assert all([isinstance(c, Node) for c in graph.roots])


def test_add_nested_children():
    graph = Graph()
    de1, de2, de3 = DummyExtractor(), DummyExtractor(), DummyExtractor()
    graph.add_nodes([de1, (de2, [(de3, [], "child's child")], 'child')])
    assert len(graph.roots) == 2
    assert isinstance(graph.roots[1].children[0], Node)
    assert graph.roots[1].children[0].name == "child's child"


def test_small_pipeline():
    pytest.importorskip('pytesseract')
    filename = join(get_test_data_path(), 'image', 'button.jpg')
    stim = ImageStim(filename)
    nodes = [(TesseractConverter(), [LengthExtractor()])]
    graph = Graph(nodes)
    result = list(graph.run([stim], merge=False))
    history = result[0].history.to_df()
    assert history.shape == (2, 8)
    assert history.iloc[0]['result_class'] == 'TextStim'
    result = merge_results(result, format='wide', extractor_names='prepend')
    assert (0, 'text[Exit]') in result['stim_name'].values
    assert 'LengthExtractor#text_length' in result.columns
    assert result['LengthExtractor#text_length'].values[0] == 4


def test_small_pipeline2():
    filename = join(get_test_data_path(), 'image', 'button.jpg')
    nodes = [BrightnessExtractor(), VibranceExtractor()]
    graph = Graph(nodes)
    result = list(graph.run([filename], merge=False))
    history = result[0].history.to_df()
    assert history.shape == (1, 8)
    result = merge_results(result, format='wide', extractor_names='multi')
    assert ('BrightnessExtractor', 'brightness') in result.columns
    brightness = result[('BrightnessExtractor', 'brightness')].values[0]
    vibrance = result[('VibranceExtractor', 'vibrance')].values[0]
    assert_almost_equal(brightness, 0.746965, 5)
    assert ('VibranceExtractor', 'vibrance') in result.columns
    assert_almost_equal(vibrance, 841.577274, 5)


def test_small_pipeline_json_spec():
    pytest.importorskip('pytesseract')
    filename = join(get_test_data_path(), 'image', 'button.jpg')
    stim = ImageStim(filename)
    nodes = {
        "roots": [
            {
                "transformer": "TesseractConverter",
                "children": [
                    {
                        "transformer": "LengthExtractor",
                        "children": []
                    }
                ]
            }
        ]
    }
    graph = Graph(nodes)
    result = list(graph.run([stim], merge=False))
    history = result[0].history.to_df()
    assert history.shape == (2, 8)
    assert history.iloc[0]['result_class'] == 'TextStim'
    result = merge_results(result, format='wide', extractor_names='multi')
    assert (0, 'text[Exit]') in result['stim_name'].values
    assert ('LengthExtractor', 'text_length') in result.columns
    assert result[('LengthExtractor', 'text_length')].values[0] == 4


def test_small_pipeline_json_spec2():
    pytest.importorskip('pytesseract')
    filename = join(get_test_data_path(), 'image', 'button.jpg')
    stim = ImageStim(filename)
    spec = join(get_test_data_path(), 'graph', 'simple_graph.json')
    graph = Graph(spec=spec)
    result = list(graph.run([stim], merge=False))
    history = result[0].history.to_df()
    assert history.shape == (2, 8)
    assert history.iloc[0]['result_class'] == 'TextStim'
    result = merge_results(result, format='wide', extractor_names='multi')
    assert (0, 'text[Exit]') in result['stim_name'].values
    assert ('LengthExtractor', 'text_length') in result.columns
    assert result[('LengthExtractor', 'text_length')].values[0] == 4


@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_small_pipeline_json_spec3():
    pytest.importorskip('pytesseract')
    filename = join(get_test_data_path(), 'image', 'button.jpg')
    stim = ImageStim(filename)
    nodes = {
        "roots": [
            {
                "transformer": "GoogleVisionAPITextConverter",
                "parameters": {
                    "num_retries": 5,
                    "max_results": 10
                },
                "children": [
                    {
                        "transformer": "LengthExtractor"
                    }
                ]
            }
        ]
    }
    graph = Graph(nodes)
    result = list(graph.run([stim], merge=False))
    history = result[0].history.to_df()
    assert history.shape == (2, 8)
    assert history.iloc[0]['result_class'] == 'TextStim'
    result = merge_results(result, format='wide', extractor_names='multi')
    assert (0, 'text[Exit\n]') in result['stim_name'].values
    assert ('LengthExtractor', 'text_length') in result.columns
    assert result[('LengthExtractor', 'text_length')].values[0] == 4


@pytest.mark.skipif("'WIT_AI_API_KEY' not in os.environ")
def test_big_pipeline():
    pytest.importorskip('pygraphviz')
    filename = join(get_test_data_path(), 'video', 'obama_speech.mp4')
    video = VideoStim(filename)
    visual_nodes = [(FrameSamplingFilter(every=15), [
        (TesseractConverter(), [LengthExtractor()]),
        VibranceExtractor(), 'BrightnessExtractor',
    ])]
    audio_nodes = [(VideoToAudioConverter(), [
        (WitTranscriptionConverter(), ['LengthExtractor'])],
        'video_to_audio')]
    graph = Graph()
    graph.add_nodes(visual_nodes)
    graph.add_nodes(audio_nodes)
    with pytest.raises(RuntimeError):
        graph.draw('temp.png')
    results = graph.run(video, merge=False)
    result = merge_results(results, format='wide', extractor_names='multi')
    # Test that pygraphviz outputs a file
    drawfile = next(tempfile._get_candidate_names())
    graph.draw(drawfile)
    graph.draw(drawfile, color=False)
    assert exists(drawfile)
    os.remove(drawfile)
    assert ('LengthExtractor', 'text_length') in result.columns
    assert ('VibranceExtractor', 'vibrance') in result.columns
    # assert not result[('onset', '')].isnull().any()
    assert 'text[negotiations]' in result['stim_name'].values
    assert 'frame[90]' in result['stim_name'].values


@pytest.mark.skipif("'WIT_AI_API_KEY' not in os.environ")
def test_big_pipeline_json():
    pytest.importorskip('pygraphviz')
    filename = join(get_test_data_path(), 'video', 'obama_speech.mp4')
    video = VideoStim(filename)
    nodes = {
        "roots": [
            {
                "transformer": "FrameSamplingFilter",
                "parameters": {
                    "every": 15
                },
                "children": [
                    {
                        "transformer": "TesseractConverter",
                        "children": [
                            {
                                "transformer": "LengthExtractor"
                            }
                        ]
                    },
                    {
                        "transformer": "VibranceExtractor"
                    },
                    {
                        "transformer": "BrightnessExtractor"
                    }
                ]
            },
            {
                "transformer": "VideoToAudioConverter",
                "children": [
                    {
                        "transformer": "WitTranscriptionConverter",
                        "children": [
                            {
                                "transformer": "LengthExtractor"
                            }
                        ]
                    }
                ]
            }
        ]
    }
    graph = Graph(nodes)
    results = graph.run(video, merge=False)
    result = merge_results(results, format='wide', extractor_names='multi')
    # Test that pygraphviz outputs a file
    drawfile = next(tempfile._get_candidate_names())
    graph.draw(drawfile)
    assert exists(drawfile)
    os.remove(drawfile)
    assert ('LengthExtractor', 'text_length') in result.columns
    assert ('VibranceExtractor', 'vibrance') in result.columns
    # assert not result[('onset', '')].isnull().any()
    assert 'text[negotiations]' in result['stim_name'].values
    assert 'frame[90]' in result['stim_name'].values


def test_stim_results():
    stim = TextStim(text='some, example the text.')
    g = Graph()
    g.add_nodes(['PunctuationRemovalFilter', 'TokenRemovalFilter',
                 'TokenizingFilter'], mode='vertical')
    final_stims = g.run(stim, merge=False)
    assert len(final_stims) == 2
    assert final_stims[1].text == 'text'

    n = Node('PunctuationRemovalFilter', name='punc')
    g = Graph([n])
    g.add_nodes(['TokenizingFilter', 'LengthExtractor'], parent=n)
    results = g.run(stim)
    assert isinstance(results, pd.DataFrame)
    assert results['LengthExtractor#text_length'][0] == 21
    with pytest.raises(ValueError):
        g.run(stim, invalid_results='fail')


@pytest.mark.skipif("'WIT_AI_API_KEY' not in os.environ")
def test_to_json():
    nodes = {
        "roots": [
            {
                "transformer": "FrameSamplingFilter",
                "parameters": {
                    "every": 15
                },
                "children": [
                    {
                        "transformer": "TesseractConverter",
                        "children": [
                            {
                                "transformer": "LengthExtractor"
                            }
                        ]
                    },
                    {
                        "transformer": "VibranceExtractor"
                    },
                    {
                        "transformer": "BrightnessExtractor"
                    }
                ]
            },
            {
                "transformer": "VideoToAudioConverter",
                "children": [
                    {
                        "transformer": "WitTranscriptionConverter",
                        "children": [
                            {
                                "transformer": "LengthExtractor"
                            }
                        ]
                    }
                ]
            }
        ]
    }
    graph = Graph(nodes)
    assert graph.to_json() == nodes
    graph = Graph(spec=join(get_test_data_path(), 'graph', 'simple_graph.json'))
    simple_graph = {
        "roots": [
            {
                "transformer": "TesseractConverter",
                "children": [
                    {
                        "transformer": "LengthExtractor"
                    }
                ]
            }
        ]
    }
    assert graph.to_json() == simple_graph
    filename = join(get_test_data_path(), 'image', 'button.jpg')
    res = graph.run(filename)
    assert res['LengthExtractor#text_length'][0] == 4


def test_save_graph():
    graph = Graph(spec=join(get_test_data_path(), 'graph', 'simple_graph.json'))
    filename = tempfile.mkstemp()[1]
    graph.save(filename)
    assert os.path.exists(filename)
    same_graph = Graph(spec=filename)
    os.remove(filename)
    assert graph.to_json() == same_graph.to_json()
    img = join(get_test_data_path(), 'image', 'button.jpg')
    res = same_graph.run(img)
    assert res['LengthExtractor#text_length'][0] == 4


def test_adding_nodes():
    graph = Graph()
    graph.add_children(['VibranceExtractor', 'BrightnessExtractor'])
    assert len(graph.roots) == 2
    assert len(graph.nodes) == 2
    for r in graph.roots:
        assert len(r.children) == 0
    img = ImageStim(join(get_test_data_path(), 'image', 'button.jpg'))
    results = graph.run(img, merge=False)
    assert len(results) == 2
    assert_almost_equal(results[0].to_df()['vibrance'][0], 841.577274, 5)
    assert_almost_equal(results[1].to_df()['brightness'][0], 0.746965, 5)

    graph = Graph()
    graph.add_chain(['PunctuationRemovalFilter', 'LengthExtractor'])
    txt = TextStim(text='the.best.text.')
    results = graph.run(txt, merge=False)
    assert len(results) == 1
    assert results[0].to_df()['text_length'][0] == 11

    with pytest.raises(ValueError):
        graph.add_nodes(['LengthExtractor'], mode='invalid')
