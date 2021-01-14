from os.path import join
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pytest
from os import environ

from ..utils import get_test_data_path
from pliers.extractors import (TensorFlowKerasApplicationExtractor,
                               TFHubExtractor, 
                               TFHubImageExtractor, 
                               TFHubTextExtractor)
from pliers.filters import AudioResamplingFilter
from pliers.stimuli import (ImageStim, 
                            TextStim, ComplexTextStim, 
                            AudioStim)
from pliers.extractors.base import merge_results


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
    eff_ext = TFHubImageExtractor(EFFNET_URL)
    eff_df = eff_ext.transform(stim).to_df()
    assert all(['feature_' + str(i) in eff_df.columns \
               for i in range(1000) ])
    assert np.argmax(np.array([eff_df['feature_' + str(i)][0] \
                     for i in range(1000)])) == 948 
    stim2 = ImageStim(join(IMAGE_DIR, 'obama.jpg'))
    mnet_ext = TFHubImageExtractor(MNET_URL, reshape_input=(224,224,3), 
                                   features='feature_vector')
    mnet_df = merge_results(mnet_ext.transform([stim, stim2]), 
                            extractor_names=False)
    assert mnet_df.shape[0] == 2
    assert all([len(v) == 1280 for v in mnet_df['feature_vector']])


def test_tfhub_text():
    stim = TextStim(join(TEXT_DIR, 'scandal.txt'))
    cstim = ComplexTextStim(join(TEXT_DIR, 'wonderful.txt'))
    sent_ext = TFHubTextExtractor(SENTENC_URL, output_key=None)
    gnews_ext = TFHubTextExtractor(GNEWS_URL, output_key=None, 
                                   features='embedding')

    sent_df = sent_ext.transform(stim).to_df()
    assert all([f'feature_{i}' in sent_df.columns for i in range(512)])
    sent_true = hub.KerasLayer(SENTENC_URL)([stim.text])[0,10].numpy()
    assert np.isclose(sent_df['feature_10'][0], sent_true)
    
    gnews_df = merge_results(gnews_ext.transform(cstim), 
                             extractor_names=False)
    assert gnews_df.shape[0] == len(cstim.elements)
    true = hub.KerasLayer(GNEWS_URL)([cstim.elements[3].text])[0,2].numpy()
    assert np.isclose(gnews_df['embedding'][3][2], true)

    with pytest.raises(ValueError) as err:
        TFHubTextExtractor(GNEWS_URL, output_key='key').transform(stim)
    assert 'not a dictionary' in str(err.value)


@pytest.mark.skipif(environ.get('CI', False) == 'true', 
                    reason='high memory')
def test_tfhub_text_transformer():
    stim = TextStim(join(TEXT_DIR, 'scandal.txt'))
    cstim = ComplexTextStim(join(TEXT_DIR, 'wonderful.txt'))
    ext = TFHubTextExtractor(ELECTRA_URL, 
                            features='sent_encoding',
                            preprocessor_url_or_path=TOKENIZER_URL)
    tkn_ext = TFHubTextExtractor(ELECTRA_URL, 
                                 features='token_encodings',
                                 output_key='sequence_output',
                                 preprocessor_url_or_path=TOKENIZER_URL)
    res = ext.transform(cstim.elements[:6])
    df = merge_results(res, extractor_names=False)
    pmod = hub.KerasLayer(TOKENIZER_URL)
    mmod = hub.KerasLayer(ELECTRA_URL)
    true = mmod(pmod([cstim.elements[5].text]))\
                ['pooled_output'][0,20].numpy()
    assert np.isclose(df['sent_encoding'][5][20], true)
    tkn_df = merge_results(tkn_ext.transform(cstim.elements[:3]), 
                           extractor_names=False)
    assert all([tkn_df['token_encodings'][i].shape == (128, 256) \
                for i in range(tkn_df.shape[0])])
    with pytest.raises(ValueError) as err:
        TFHubTextExtractor(ELECTRA_URL, 
                           preprocessor_url_or_path=TOKENIZER_URL,
                           output_key='key').transform(stim)
    assert 'Check which keys' in str(err.value)


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
