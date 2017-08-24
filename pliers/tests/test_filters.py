from os.path import join
from .utils import get_test_data_path
from pliers.filters import (WordStemmingFilter,
                            ImageCroppingFilter)
from pliers.stimuli import ComplexTextStim, ImageStim
from nltk import stem as nls
import numpy as np
import pytest


TEXT_DIR = join(get_test_data_path(), 'text')
IMAGE_DIR = join(get_test_data_path(), 'image')


def test_word_stemming_filter():
    stim = ComplexTextStim(join(TEXT_DIR, 'sample_text.txt'),
                           columns='to', default_duration=1)

    # With all defaults (porter stemmer)
    filt = WordStemmingFilter()
    assert isinstance(filt.stemmer, nls.PorterStemmer)
    stemmed = filt.transform(stim)
    stems = [s.text for s in stemmed]
    target = ['some', 'sampl', 'text', 'for', 'test', 'annot']
    assert stems == target

    # Try a different stemmer
    filt = WordStemmingFilter(stemmer='snowball', language='english')
    assert isinstance(filt.stemmer, nls.SnowballStemmer)
    stemmed = filt.transform(stim)
    stems = [s.text for s in stemmed]
    assert stems == target

    # Handles StemmerI stemmer
    stemmer = nls.SnowballStemmer(language='english')
    filt = WordStemmingFilter(stemmer=stemmer)
    stemmed = filt.transform(stim)
    stems = [s.text for s in stemmed]
    assert stems == target

    # Fails on invalid values
    with pytest.raises(ValueError):
        filt = WordStemmingFilter(stemmer='nonexistent_stemmer')


def test_image_cropping_filter():
    stim = ImageStim(join(IMAGE_DIR, 'apple.jpg'))
    filt = ImageCroppingFilter((210, 120, 260, 170))
    new_stim = filt.transform(stim)
    assert new_stim.data.shape == (50, 50, 3)
    assert np.array_equal(stim.data[0, 0], [255.0, 255.0, 255.0])
    # Top left corner goes white -> red
    assert np.array_equal(new_stim.data[0, 0], [136.0, 0.0, 0.0])
