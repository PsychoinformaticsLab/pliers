'''
Extractors that operate primarily or exclusively on Text stimuli.
'''
import sys
#sys.path.insert(0,'C:\\Users\\jayee\\Documents\\Tal-Pliers\\pliers' )
from pliers.stimuli.text import TextStim, ComplexTextStim
from pliers.extractors.base import Extractor, ExtractorResult
from pliers.support.exceptions import PliersError
from pliers.support.decorators import requires_nltk_corpus
from pliers.datasets.text import fetch_dictionary
from pliers.transformers import BatchTransformerMixin
from pliers.utils import attempt_to_import, verify_dependencies
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import re
import logging
from six import string_types




keyedvectors = attempt_to_import('gensim.models.keyedvectors', 'keyedvectors',
                                 ['KeyedVectors'])
sklearn_text = attempt_to_import('sklearn.feature_extraction.text', 'sklearn_text',
                                 ['VectorizerMixin', 'CountVectorizer'])
spacy = attempt_to_import('spacy')

class TextExtractor(Extractor):

    ''' Base Text Extractor class; all subclasses can only be applied to text.
    '''
    _input_type = TextStim


class ComplexTextExtractor(Extractor):

    ''' Base ComplexTextStim Extractor class; all subclasses can only be
    applied to ComplexTextStim instance.
    '''
    _input_type = ComplexTextStim

    def _extract(self, stim):
        ''' Returns all words. '''
        props = [(e.text, e.onset, e.duration) for e in stim.elements]
        vals, onsets, durations = map(list, zip(*props))
        return ExtractorResult(vals, stim, self, ['word'], onsets, durations)


class DictionaryExtractor(TextExtractor):

    ''' A generic dictionary-based extractor that supports extraction of
    arbitrary features contained in a lookup table.

    Args:
        dictionary (str, DataFrame): The dictionary containing the feature
            values. Either a string giving the path to the dictionary file,
            or a pandas DF. Format must be tab-delimited, with the first column
            containing the text key used for lookup. Subsequent columns each
            represent a single feature that can be used in extraction.
        variables (list): Optional subset of columns to keep from the
            dictionary.
        missing: Value to insert if no lookup value is found for a text token.
            Defaults to numpy's NaN.
    '''

    _log_attributes = ('dictionary', 'variables', 'missing')
    VERSION = '1.0'

    def __init__(self, dictionary, variables=None, missing=np.nan):
        if isinstance(dictionary, string_types):
            self.dictionary = dictionary  # for TranformationHistory logging
            dictionary = pd.read_csv(dictionary, sep='\t', index_col=0)
        else:
            self.dictionary = None
        self.data = dictionary
        if variables is None:
            variables = list(self.data.columns)
        else:
            self.data = self.data[variables]
        self.variables = variables
        # Set up response when key is missing
        self.missing = missing
        super(DictionaryExtractor, self).__init__()

    def _extract(self, stim):
        if stim.text not in self.data.index:
            vals = pd.Series(self.missing, self.variables)
        else:
            vals = self.data.loc[stim.text].fillna(self.missing)
        vals = vals.to_dict()
        return ExtractorResult(np.array([list(vals.values())]), stim, self,
                               features=list(vals.keys()))


class PredefinedDictionaryExtractor(DictionaryExtractor):

    ''' A generic Extractor that maps words onto values via one or more
    pre-defined dictionaries accessed via the web.

    Args:
        variables (list or dict): A specification of the dictionaries and
            column names to map the input TextStims onto. If a list, each
            element must be a string with the format 'dict/column', where the
            value before the slash gives the name of the dictionary, and the
            value after the slash gives the name of the column in that
            dictionary. These names can be found in the dictionaries.json
            specification file under the datasets submodule. Examples of
            valid values are 'affect/V.Mean.Sum' and
            'subtlexusfrequency/Lg10WF'. If a dict, the keys are the names of
            the dictionary files (e.g., 'affect'), and the values are lists
            of columns to use (e.g., ['V.Mean.Sum', 'V.SD.Sum']).
        missing (object): Value to use when an entry for a word is missing in
            a dictionary (defaults to numpy's NaN).
        case_sensitive (bool): If True, entries in the dictionary are treated
            as case-sensitive (e.g., 'John' and 'john' are different words).
    '''

    _log_attributes = ('variables', 'missing', 'case_sensitive')
    VERSION = '1.0'

    def __init__(self, variables, missing=np.nan, case_sensitive=True):

        self.case_sensitive = case_sensitive

        if isinstance(variables, (list, tuple)):
            _vars = {}
            for v in variables:
                v = v.split('/')
                if v[0] not in _vars:
                    _vars[v[0]] = []
                if len(v) == 2:
                    _vars[v[0]].append(v[1])
            variables = _vars

        dicts = []
        for k, v in variables.items():
            d = fetch_dictionary(k)
            if not case_sensitive:
                d.index = d.index.str.lower()
            if v:
                d = d[v]
            d.columns = ['%s_%s' % (k, c) for c in d.columns]
            dicts.append(d)

        dictionary = pd.concat(dicts, axis=1, join='outer')
        super(PredefinedDictionaryExtractor, self).__init__(
            dictionary, missing=missing)


class LengthExtractor(TextExtractor):

    ''' Extracts the length of the text in characters. '''

    VERSION = '1.0'

    def _extract(self, stim):
        return ExtractorResult(np.array([[len(stim.text.strip())]]), stim,
                               self, features=['text_length'])


class NumUniqueWordsExtractor(TextExtractor):

    ''' Extracts the number of unique words used in the text. '''

    _log_attributes = ('tokenizer',)
    VERSION = '1.0'

    def __init__(self, tokenizer=None):
        super(NumUniqueWordsExtractor, self).__init__()
        self.tokenizer = tokenizer

    @requires_nltk_corpus
    def _extract(self, stim):
        text = stim.text
        if self.tokenizer is None:
            if nltk is None:
                num_words = len(set(text.split()))
            else:
                num_words = len(set(nltk.word_tokenize(text)))
        else:
            num_words = len(set(self.tokenizer.tokenize(text)))

        return ExtractorResult(np.array([[num_words]]), stim, self,
                               features=['num_unique_words'])


class PartOfSpeechExtractor(BatchTransformerMixin, TextExtractor):

    ''' Tags parts of speech in text with nltk. '''

    _batch_size = sys.maxsize
    VERSION = '1.0'

    @requires_nltk_corpus
    def _extract(self, stims):
        words = [w.text for w in stims]
        pos = nltk.pos_tag(words)
        if len(words) != len(pos):
            raise PliersError(
                "The number of words does not match the number of tagged words"
                "returned by nltk's part-of-speech tagger.")

        results = []
        tagset = nltk.data.load('help/tagsets/upenn_tagset.pickle').keys()
        for i, s in enumerate(stims):
            pos_vector = dict.fromkeys(tagset, 0)
            pos_vector[pos[i][1]] = 1
            values = [list(pos_vector.values())]
            results.append(ExtractorResult(values, s, self,
                                           features=list(pos_vector.keys())))

        return results


class WordEmbeddingExtractor(TextExtractor):

    ''' An extractor that uses a word embedding file to look up embedding
    vectors for text.

    Args:
        embedding_file (str): Path to a word embedding file. Assumed to be in
            word2vec format compatible with gensim.
        binary (bool): Flag indicating whether embedding file is saved in a
            binary format.
        prefix (str): Prefix for feature names in the ExtractorResult.
        unk_vector (numpy array or str): Default vector to use for texts not
            found in the embedding file. If None is specified, uses a
            vector with all zeros. If 'random' is specified, uses a vector with
            random values between -1.0 and 1.0. Must have the same dimensions
            as the embeddings.
    '''

    _log_attributes = ('wvModel', 'prefix')

    def __init__(self, embedding_file, binary=False, prefix='embedding_dim',
                 unk_vector=None):
        verify_dependencies(['keyedvectors'])
        self.wvModel = keyedvectors.KeyedVectors.load_word2vec_format(embedding_file, binary=binary)
        self.prefix = prefix
        self.unk_vector = unk_vector
        super(WordEmbeddingExtractor, self).__init__()

    def _extract(self, stim):
        num_dims = self.wvModel.vector_size
        if stim.text in self.wvModel:
            embedding_vector = self.wvModel[stim.text]
        else:
            unk = self.unk_vector
            if hasattr(unk, 'shape') and unk.shape[0] == num_dims:
                embedding_vector = unk
            elif unk == 'random':
                embedding_vector = 2.0 * np.random.random(num_dims) - 1.0
            else:
                # By default, UNKs will have zeroed-out vectors
                embedding_vector = np.zeros(num_dims)

        features = ['%s%d' % (self.prefix, i) for i in range(num_dims)]
        return ExtractorResult([embedding_vector],
                               stim,
                               self,
                               features=features)


class TextVectorizerExtractor(BatchTransformerMixin, TextExtractor):

    ''' Uses a scikit-learn Vectorizer to extract bag-of-features
    from text.

    Args:
        vectorizer (sklearn Vectorizer or str): a scikit-learn Vectorizer
            (or the name in a string) to extract with. Will use the
            CountVectorizer by default. Uses supporting *args and **kwargs.
    '''

    _log_attributes = ('vectorizer',)
    _batch_size = sys.maxsize

    def __init__(self, vectorizer=None, *vectorizer_args, **vectorizer_kwargs):
        verify_dependencies(['sklearn_text'])
        if isinstance(vectorizer, sklearn_text.VectorizerMixin):
            self.vectorizer = vectorizer
        elif isinstance(vectorizer, str):
            vec = getattr(sklearn_text, vectorizer)
            self.vectorizer = vec(*vectorizer_args, **vectorizer_kwargs)
        else:
            self.vectorizer = sklearn_text.CountVectorizer(*vectorizer_args,
                                                           **vectorizer_kwargs)
        super(TextVectorizerExtractor, self).__init__()

    def _extract(self, stims):
        mat = self.vectorizer.fit_transform([s.text for s in stims]).toarray()
        results = []
        for i, row in enumerate(mat):
            results.append(
                ExtractorResult([row], stims[i], self,
                                features=self.vectorizer.get_feature_names()))
        return results


class VADERSentimentExtractor(TextExtractor):

    ''' Uses nltk's VADER lexicon to extract (0.0-1.0) values for the positve,
    neutral, and negative sentiment of a TextStim. Also returns a compound
    score ranging from -1 (very negative) to +1 (very positive). '''

    _log_attributes = ('analyzer',)
    VERSION = '1.0'

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        super(VADERSentimentExtractor, self).__init__()

    @requires_nltk_corpus
    def _extract(self, stim):
        scores = self.analyzer.polarity_scores(stim.text)
        features = ['sentiment_' + k for k in scores.keys()]
        return ExtractorResult([list(scores.values())], stim, self,
                               features=features)



class SpaCyExtractor(TextExtractor):
    
    ''' A generic class for Spacy Text extractors 
    
   
    Uses SpaCy to extract features from text. Extracts features for every word
    (token) in a sentence.
    
    Args:
        extractor_type(str): The type of feature to extract. Must be one of
            'doc' (analyze an entire sentence/document) or 'token'
            (analyze each word).
        features(list): A list of strings giving the names of spaCy features to
            extract. See SpacY documentation for details. By default, returns
            all available features for the given extractor type.
        model (str): The name of the language model to use.
    '''
    
    def __init__(self, extractor_type='token', features=None,
                 model='en_core_web_sm'):

        verify_dependencies(['spacy'])

        try:
            self.model = spacy.load(model)
        except (ImportError, IOError, OSError) as e:
            logging.warning("Spacy Models ('{}') not found. Downloading and"
                            "installing".format(model))

            os.system('python -m spacy download {}'.format(model))
            self.model = spacy.load(model)

        logging.info('Loaded model: {}'.format(self.model))

        self.features = features
        self.extractor_type = extractor_type.lower()

        super(SpaCyExtractor, self).__init__()
        
    def _extract(self, stim):
   
        features_list = []
        elements = self.model(stim.text)
        order_list = []
        
        if self.extractor_type == 'token':
            if self.features is None:
                self.features = ['text', 'lemma_', 'pos_', 'tag_', 'dep_',
                                 'shape_', 'is_alpha', 'is_stop', 'is_punct',
                                 'sentiment', 'is_ascii', 'is_digit']

        elif self.extractor_type == 'doc':
             elements = [elem.as_doc() for elem in list(elements.sents)]
             if self.features is None:
                 self.features = ['text', 'is_tagged', 'is_parsed',
                                  'is_sentenced', 'sentiment']

        else:
            raise(ValueError("Invalid extractor_type; must be one of 'token'"
                             " or 'doc'."))

        features_list = []
        for elem in elements:
            arr = []
            for feat in self.features:
                arr.append(getattr(elem, feat))
            features_list.append(arr)

        order_list = list(range(1, len(elements) + 1))

        return ExtractorResult(features_list, stim, self,
                               features=self.features, orders=order_list)
