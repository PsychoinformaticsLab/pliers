from os.path import join
import tensorflow_hub as hub
import numpy as np
import pytest

from ..utils import get_test_data_path
from pliers.extractors import (TensorFlowKerasApplicationExtractor,
                               TFHubExtractor, TFHubImageExtractor, 
                               TFHubTextExtractor)
from pliers.stimuli import ImageStim, TextStim, ComplexTextStim
from pliers.extractors.base import merge_results


IMAGE_DIR = join(get_test_data_path(), 'image')
TEXT_DIR = join(get_test_data_path(), 'text')

EFFICIENTNET_URL = 'https://tfhub.dev/tensorflow/efficientnet/b7/classification/1'
INCEPTION_URL = 'https://tfhub.dev/google/imagenet/inception_v2/feature_vector/4'

SENTENCE_ENCODER_URL = 'https://tfhub.dev/google/universal-sentence-encoder/4'
GOOGLE_NEWS_URL = 'https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2'
ELECTRA_PREPROCESSOR_URL = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2'
ELECTRA_MODEL_URL = 'https://tfhub.dev/google/electra_small/2'

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
    # test two models
    # test reshape warning and reshape
    # test postprocessing 

def test_tfhub_text():
    txt = TextStim(join(TEXT_DIR, 'scandal.txt'))
    ctxt = ComplexTextStim(join(TEXT_DIR, 'wonderful.txt'))
    sent_ext = TFHubTextExtractor(SENTENCE_ENCODER_URL, 
                                  output_key=None,)
    gnews_ext = TFHubTextExtractor(GOOGLE_NEWS_URL,
                                   output_key=None, features='embedding')
    electra_ext = TFHubTextExtractor(ELECTRA_MODEL_URL, 
                                     preprocessor_url_or_path=ELECTRA_PREPROCESSOR_URL)
    
    sent_df = sent_ext.transform(txt).to_df()
    assert all([f'feature_{i}' in sent_df.columns for i in range(512)])
    assert np.isclose(sent_df['feature_10'][0], 
                      hub.KerasLayer(SENTENCE_ENCODER_URL)([txt.text])[0,10].numpy())
    
    gnews_df = merge_results(gnews_ext.transform(ctxt), extractor_names=False)
    assert gnews_df.shape[0] == len(ctxt.elements)
    assert np.isclose(gnews_df['embedding'][3][2],
                      hub.KerasLayer(GOOGLE_NEWS_URL)([ctxt.elements[3].text])[0,2].numpy())
    
    electra_ext = electra_ext.transform(ctxt)
    # test preprocessing step
    # test transformations

def test_tfhub_generic():
    pass