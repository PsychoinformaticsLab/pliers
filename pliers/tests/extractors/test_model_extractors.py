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
                               TFHubTextExtractor,
                               BertExtractor,
                               BertSequenceEncodingExtractor,
                               BertLMExtractor,
                               BertSentimentExtractor,
                               AudiosetLabelExtractor)
from pliers.filters import AudioResamplingFilter
from pliers.stimuli import (ImageStim,
                            TextStim, ComplexTextStim,
                            AudioStim)
from pliers.extractors.base import merge_results
from transformers import BertTokenizer
from pliers.utils import verify_dependencies


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


pytestmark = pytest.mark.skipif(
    environ.get('skip_high_memory', False) == 'true', reason='high memory')


def test_tensorflow_keras_application_extractor():
    imgs = [join(IMAGE_DIR, f) for f in ['apple.jpg', 'obama.jpg']]
    imgs = [ImageStim(im, onset=4.2, duration=1) for im in imgs]
    ext = TensorFlowKerasApplicationExtractor()
    results = ext.transform(imgs)
    df = merge_results(results, format='wide', extractor_names='multi')
    assert df.shape == (2, 19)
    true = 0.9737075
    pred = df['TensorFlowKerasApplicationExtractor'].loc[0, 'Granny_Smith']
    assert np.isclose(true, pred, 1e-02)
    true = 0.64234024
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


def test_tfhub_text():
    stim = TextStim(join(TEXT_DIR, 'scandal.txt'))
    ext = TFHubTextExtractor(SENTENC_URL, output_key=None)
    df = ext.transform(stim).to_df()
    assert all([f'feature_{i}' in df.columns for i in range(512)])
    true = hub.KerasLayer(SENTENC_URL)([stim.text])[0,10].numpy()
    assert np.isclose(df['feature_10'][0], true)


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


def test_tfhub_text_transformer_sentence():
    stim = TextStim(join(TEXT_DIR, 'scandal.txt'))
    cstim = ComplexTextStim(join(TEXT_DIR, 'wonderful.txt'))
    ext = TFHubTextExtractor(ELECTRA_URL,
                            features='sent_encoding',
                            preprocessor_url_or_path=TOKENIZER_URL)
    res = ext.transform(cstim.elements[:6])
    df = merge_results(res, extractor_names=False)
    pmod = hub.KerasLayer(TOKENIZER_URL)
    mmod = hub.KerasLayer(ELECTRA_URL)
    true = mmod(pmod([cstim.elements[5].text]))\
                ['pooled_output'][0,20].numpy()
    assert np.isclose(df['sent_encoding'][5][20], true)
    with pytest.raises(ValueError) as err:
        TFHubTextExtractor(ELECTRA_URL,
                           preprocessor_url_or_path=TOKENIZER_URL,
                           output_key='key').transform(stim)
    assert 'Check which keys' in str(err.value)


def test_tfhub_text_transformer_tokens():
    cstim = ComplexTextStim(join(TEXT_DIR, 'wonderful.txt'))
    tkn_ext = TFHubTextExtractor(ELECTRA_URL,
                                 features='token_encodings',
                                 output_key='sequence_output',
                                 preprocessor_url_or_path=TOKENIZER_URL)
    tkn_df = merge_results(tkn_ext.transform(cstim.elements[:3]),
                           extractor_names=False)
    assert all([tkn_df['token_encodings'][i].shape == (128, 256) \
                for i in range(tkn_df.shape[0])])


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


def test_bert_extractor():
    stim = ComplexTextStim(text='This is not a tokenized sentence.')
    stim_file = ComplexTextStim(join(TEXT_DIR, 'sentence_with_header.txt'))

    ext_base = BertExtractor(pretrained_model='bert-base-uncased')
    ext_base_token = BertExtractor(pretrained_model='bert-base-uncased',
                                   return_input=True)
    ext_tf = BertExtractor(pretrained_model='bert-base-uncased', framework='tf')

    base_result = ext_base.transform(stim)
    res = base_result.to_df()
    res_token = ext_base_token.transform(stim).to_df()
    res_file = ext_base.transform(stim_file).to_df()
    res_tf = ext_tf.transform(stim).to_df()

    # Test encoding shape
    assert len(res['encoding'][0]) == 768
    assert len(res_file['encoding'][0]) == 768

    # test base extractor
    assert res.shape[0] == 8
    assert res_token.shape[0] == 8
    assert res_token['token'][5] == '##ized'
    assert res_token['word'][5] == 'tokenized'
    assert res_token['object_id'][5] == 5

    # test base extractor on file
    assert res_file.shape[0] == 8
    assert res_file['onset'][3] == 1.3
    assert res_file['duration'][5] == 0.5
    assert res_file['object_id'][5] == 5

    # test tf vs torch
    cors = [np.corrcoef(res['encoding'][i], res_tf['encoding'][i])[0,1]
            for i in range(res.shape[0])]
    assert all(np.isclose(cors, 1))

    # catch error if framework is invalid
    with pytest.raises(ValueError) as err:
        BertExtractor(framework='keras')
    assert 'Invalid framework' in str(err.value)

    # Delete the models
    del res, res_token, res_file, ext_base, ext_base_token


@pytest.mark.parametrize('model',
                         ['bert-large-uncased', 'distilbert-base-uncased',
                          'roberta-base', 'camembert-base'])
def test_bert_other_models(model):
    if model == 'camembert-base':
        stim = ComplexTextStim(text='ceci n\'est pas un pipe')
    else:
        stim = ComplexTextStim(text='This is not a tokenized sentence.')
    res = BertExtractor(
        pretrained_model=model, return_input=True).transform(stim).to_df()
    if model == 'bert-large-uncased':
        shape = 1024
    else:
        shape = 768
    assert len(res['encoding'][0]) == shape
    if model == 'camembert-base':
        assert res['token'][4] == 'est'

    # remove variables
    del res, stim


def test_bert_sequence_extractor():
    stim = ComplexTextStim(text='This is not a tokenized sentence.')
    stim_file = ComplexTextStim(join(TEXT_DIR, 'sentence_with_header.txt'))

    ext_pooler = BertSequenceEncodingExtractor(return_special='pooler_output')

    # Test correct behavior when setting return_special
    assert ext_pooler.pooling is None
    assert ext_pooler.return_special == 'pooler_output'

    res_sequence = BertSequenceEncodingExtractor(
        return_input=True).transform(stim).to_df()
    res_file = BertSequenceEncodingExtractor(
        return_input=True).transform(stim_file).to_df()
    res_cls = BertSequenceEncodingExtractor(
        return_special='[CLS]').transform(stim).to_df()
    res_pooler = ext_pooler.transform(stim).to_df()
    res_max = BertSequenceEncodingExtractor(
        pooling='max').transform(stim).to_df()

    # Check shape
    assert len(res_sequence['encoding'][0]) == 768
    assert len(res_cls['encoding'][0]) == 768
    assert len(res_pooler['encoding'][0]) == 768
    assert len(res_max['encoding'][0]) == 768
    assert res_sequence.shape[0] == 1
    assert res_cls.shape[0] == 1
    assert res_pooler.shape[0] == 1
    assert res_max.shape[0] == 1

    # Make sure pooler/cls/no arguments return different encodings
    assert res_sequence['encoding'][0] != res_cls['encoding'][0]
    assert res_sequence['encoding'][0] != res_pooler['encoding'][0]
    assert res_sequence['encoding'][0] != res_max['encoding'][0]
    assert all([res_max['encoding'][0][i] >= res_sequence['encoding'][0][i]
                                              for i in range(768)])

    # test return sequence
    assert res_sequence['sequence'][0] == 'This is not a tokenized sentence .'

    # test file stim
    assert res_file['duration'][0] == 2.9
    assert res_file['onset'][0] == 0.2

    # catch error with wrong numpy function and wrong special token arg
    with pytest.raises(ValueError) as err:
        BertSequenceEncodingExtractor(pooling='avg')
    assert 'valid numpy function' in str(err.value)
    with pytest.raises(ValueError) as err:
        BertSequenceEncodingExtractor(return_special='[MASK]')
    assert 'must be one of' in str(err.value)

    # remove variables
    del ext_pooler, res_cls, res_max, res_pooler, res_sequence, res_file, stim


def test_bert_LM_extractor():
    stim = ComplexTextStim(text='This is not a tokenized sentence.')
    stim_masked = ComplexTextStim(text='This is MASK tokenized sentence.')
    stim_file = ComplexTextStim(join(TEXT_DIR, 'sentence_with_header.txt'))

    # Test mutual exclusivity and mask values
    with pytest.raises(ValueError) as err:
        BertLMExtractor(top_n=100, target='test')
    assert 'mutually exclusive' in str(err.value)
    with pytest.raises(ValueError) as err:
        BertLMExtractor(top_n=100, threshold=.5)
    assert 'mutually exclusive' in str(err.value)
    with pytest.raises(ValueError) as err:
        BertLMExtractor(target='test', threshold=.5)
    assert 'mutually exclusive' in str(err.value)
    with pytest.raises(ValueError) as err:
        BertLMExtractor(mask=['test', 'mask'])
    assert 'must be a string' in str(err.value)
    with pytest.raises(ValueError) as err:
        BertLMExtractor(target='nonwd')
    assert 'No valid target token' in str(err.value)

    target_wds = ['target', 'word']
    ext_target = BertLMExtractor(mask=1, target=target_wds)

    res = BertLMExtractor(mask=2).transform(stim).to_df()
    res_file = BertLMExtractor(mask=2).transform(stim_file).to_df()
    res_target = ext_target.transform(stim).to_df()
    res_topn = BertLMExtractor(mask=3, top_n=100).transform(stim).to_df()
    res_threshold = BertLMExtractor(
        mask=4, threshold=.1, return_softmax=True).transform(stim).to_df()
    res_default = BertLMExtractor().transform(stim_masked).to_df()
    res_return_mask = BertLMExtractor(
        mask=1, top_n=10, return_masked_word=True, return_input=True).transform(stim).to_df()

    assert res.shape[0] == 1

    # test onset/duration
    assert res_file['onset'][0] == 1.0
    assert res_file['duration'][0] == 0.2

    # Check target words
    assert all([w.capitalize() in res_target.columns for w in target_wds])
    assert res_target.shape[1] == 6

    # Check top_n
    assert res_topn.shape[1] == 104
    assert all([res_topn.iloc[:, 3][0] > res_topn.iloc[:, i][0] for i in range(4, 103)])

    # Check threshold and range
    tknz = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tknz.vocab.keys()
    for v in vocab:
        if v.capitalize() in res_threshold.columns:
            assert res_threshold[v.capitalize()][0] >= .1
            assert res_threshold[v.capitalize()][0] <= 1

    # Test update mask method
    assert ext_target.mask == 1
    ext_target.update_mask(new_mask='sentence')
    assert ext_target.mask == 'sentence'
    res_target_new = ext_target.transform(stim).to_df()
    assert all([res_target[c][0] != res_target_new[c][0]
                for c in ['Target', 'Word']])
    with pytest.raises(ValueError) as err:
        ext_target.update_mask(new_mask=['some', 'mask'])
    assert 'must be a string' in str(err.value)

    # Test default mask
    assert res_default.shape[0] == 1

    # Test return mask and input
    assert res_return_mask['true_word'][0] == 'is'
    assert 'true_word_score' in res_return_mask.columns
    assert res_return_mask['sequence'][0] == 'This is not a tokenized sentence .'

    # Make sure no non-ascii tokens are dropped
    assert res.shape[1] == len(vocab) + 4

    # remove variables
    del ext_target, res, res_file, res_target, res_topn, \
        res_threshold, res_default, res_return_mask


def test_bert_sentiment_extractor():
    stim = ComplexTextStim(text='This is the best day of my life.')
    stim_file = ComplexTextStim(join(TEXT_DIR, 'sentence_with_header.txt'))

    res = BertSentimentExtractor().transform(stim).to_df()
    res_file = BertSentimentExtractor().transform(stim_file).to_df()
    res_seq = BertSentimentExtractor(return_input=True).transform(stim).to_df()
    res_softmax = BertSentimentExtractor(
        return_softmax=True).transform(stim).to_df()

    assert res.shape[0] == 1
    assert res_file['onset'][0] == 0.2
    assert res_file['duration'][0] == 2.9
    assert all([s in res.columns for s in ['sent_pos', 'sent_neg']])
    assert res_seq['sequence'][0] == 'This is the best day of my life .'
    assert all([res_softmax[s][0] >= 0 for s in ['sent_pos', 'sent_neg']])
    assert all([res_softmax[s][0] <= 1 for s in ['sent_pos', 'sent_neg']])

    # remove variables
    del res, res_file, res_seq, res_softmax


@pytest.mark.parametrize('hop_size', [0.1, 1])
@pytest.mark.parametrize('top_n', [5, 10])
@pytest.mark.parametrize('target_sr', [22000, 14000])
def test_audioset_extractor(hop_size, top_n, target_sr):
    verify_dependencies(['tensorflow'])

    def compute_expected_length(stim, ext):
        stft_par = ext.params.STFT_WINDOW_SECONDS - ext.params.STFT_HOP_SECONDS
        tot_window = ext.params.PATCH_WINDOW_SECONDS + stft_par
        ons = np.arange(
            start=0, stop=stim.duration - tot_window, step=hop_size)
        return len(ons)

    audio_stim = AudioStim(join(AUDIO_DIR, 'crowd.mp3'))
    audio_filter = AudioResamplingFilter(target_sr=target_sr)
    audio_resampled = audio_filter.transform(audio_stim)

    # test with defaults and 44100 stimulus
    ext = AudiosetLabelExtractor(hop_size=hop_size)
    r_orig = ext.transform(audio_stim).to_df()
    assert r_orig.shape[0] == compute_expected_length(audio_stim, ext)
    assert r_orig.shape[1] == 525
    assert np.argmax(r_orig.to_numpy()[:, 4:].mean(axis=0)) == 0
    assert r_orig['duration'][0] == .975
    assert all([np.isclose(r_orig['onset'][i] - r_orig['onset'][i-1], hop_size)
                for i in range(1, r_orig.shape[0])])

    # test resampled audio length and errors
    if target_sr >= 14500:
        r_resampled = ext.transform(audio_resampled).to_df()
        assert r_orig.shape[0] == r_resampled.shape[0]
    else:
        with pytest.raises(ValueError) as sr_error:
            ext.transform(audio_resampled)
        assert all([substr in str(sr_error.value)
                    for substr in ['Upsample', str(target_sr)]])

    # test top_n option
    ext_top_n = AudiosetLabelExtractor(top_n=top_n)
    r_top_n = ext_top_n.transform(audio_stim).to_df()
    assert r_top_n.shape[1] == ext_top_n.top_n + 4
    assert np.argmax(r_top_n.to_numpy()[:, 4:].mean(axis=0)) == 0

    # test label subset
    labels = ['Speech', 'Silence', 'Harmonic', 'Bark', 'Music', 'Bell',
              'Steam', 'Rain']
    ext_labels_only = AudiosetLabelExtractor(labels=labels)
    r_labels_only = ext_labels_only.transform(audio_stim).to_df()
    assert r_labels_only.shape[1] == len(labels) + 4

    # test top_n/labels error
    with pytest.raises(ValueError) as err:
        AudiosetLabelExtractor(top_n=10, labels=labels)
    assert 'Top_n and labels are mutually exclusive' in str(err.value)
