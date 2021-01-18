from os.path import join
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pytest
from os import environ
from pliers.tests.utils import get_test_data_path
from pliers import config
from pliers.extractors import (TensorFlowKerasApplicationExtractor,
                               TFHubExtractor,
                               TFHubImageExtractor,
                               TFHubTextExtractor)
from pliers.filters import AudioResamplingFilter
from pliers.stimuli import (ImageStim,
                            TextStim, ComplexTextStim,
                            AudioStim)
from pliers.extractors.base import merge_results

cache_default = config.get_option('cache_transformers')
config.set_option('cache_transformers', False)

IMAGE_DIR = join(get_test_data_path(), 'image')
TEXT_DIR = join(get_test_data_path(), 'text')
AUDIO_DIR = join(get_test_data_path(), 'audio')

EFFNET_URL = 'https://tfhub.dev/tensorflow/efficientnet/b7/classification/1'
MNET_URL = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4'
SENTENC_URL = 'https://tfhub.dev/google/universal-sentence-encoder/4'
GNEWS_URL = 'https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2'
TOKENIZER_URL = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2'
ELECTRA_URL = 'https://tfhub.dev/google/electra_small/2'
SPEECH_URL = 'https://tfhub.dev/google/speech_embedding/1'


def test_tensorflow_keras_application_extractor():
    imgs = [join(IMAGE_DIR, f) for f in ['apple.jpg', 'obama.jpg']]
    imgs = [ImageStim(im, onset=4.2, duration=1) for im in imgs]
    ext = TensorFlowKerasApplicationExtractor()
    results = ext.transform(imgs)
    df = merge_results(results, format='wide', extractor_names='multi')
    assert df.shape == (2, 19)
    true = 0.9737075
    pred = df['TensorFlowKerasApplicationExtractor'].loc[0, 'Granny_Smith']
    assert np.isclose(true, pred, 1e-05)
    true = 0.64234024
    pred = df['TensorFlowKerasApplicationExtractor'].loc[1, 'Windsor_tie']
    assert np.isclose(true, pred, 1e-05)
    assert 4.2 in df[('onset', np.nan)].values
    assert 1 in df[('duration', np.nan)].values
    with pytest.raises(ValueError):
        TensorFlowKerasApplicationExtractor(architecture='foo')


def test_tfhub_image():
    stim = ImageStim(join(IMAGE_DIR, 'apple.jpg'))
    ext = TFHubImageExtractor(EFFNET_URL)
    df = ext.transform(stim).to_df()
    assert all(['feature_' + str(i) in df.columns \
               for i in range(1000) ])
    assert np.argmax(np.array([df['feature_' + str(i)][0] \
                     for i in range(1000)])) == 948


def test_tfhub_image_reshape():
    stim = ImageStim(join(IMAGE_DIR, 'apple.jpg'))
    stim2 = ImageStim(join(IMAGE_DIR, 'obama.jpg'))
    ext = TFHubImageExtractor(MNET_URL, 
                              reshape_input=(224,224,3),
                              features='feature_vector')
    df = merge_results(ext.transform([stim, stim2]),
                       extractor_names=False)
    assert df.shape[0] == 2
    assert all([len(v) == 1280 for v in df['feature_vector']])


def test_tfhub_text_one_feature():
    stim = TextStim(join(TEXT_DIR, 'scandal.txt'))
    cstim = ComplexTextStim(join(TEXT_DIR, 'wonderful.txt'))
    ext = TFHubTextExtractor(GNEWS_URL, output_key=None,
                                   features='embedding')
    df = merge_results(ext.transform(cstim), extractor_names=False)
    assert df.shape[0] == len(cstim.elements)
    true = hub.KerasLayer(GNEWS_URL)([cstim.elements[3].text])[0,2].numpy()
    assert np.isclose(df['embedding'][3][2], true)
    with pytest.raises(ValueError) as err:
        TFHubTextExtractor(GNEWS_URL, output_key='key').transform(stim)
    assert 'not a dictionary' in str(err.value)


def test_tfhub_generic():
    # Test generic extractor with speech embedding model
    astim = AudioStim(join(AUDIO_DIR, 'obama_speech.wav'))
    astim = AudioResamplingFilter(target_sr=16000).transform(astim)
    transform_fn = lambda x: tf.expand_dims(x, axis=0)
    aext = TFHubExtractor(SPEECH_URL,
                          transform_inp=transform_fn,
                          features='speech_embedding')
    df = aext.transform(astim).to_df()
    # Check expected dimensionality (see model URL)
    emb_dim = 96
    n_chunks = 1 + (astim.data.shape[0] - 12400) // 1280
    assert df['speech_embedding'][0].shape == (n_chunks, emb_dim)
