# Changelog

## 0.2 (January 8, 2018)
This is a major release that adds many new features and transformers. Changes include:
* Sphinx [documentation](http://tyarkoni.github.io/pliers/)!
* New package features:
    * Ability to create transformer Graphs via a JSON specification
    * Scitkit-learn integration: ability to use pliers Graphs or Extractors as sklearn Transformers that can be plugged into full Pipelines
    * Batch transformation: added ability to transform stimuli in batches when the transformer inherits `BatchTransformerMixin`, but a batch size must be given
    * Static versioning for transformers (0.1 indicates initial working transformer, 1.0+ indicates fully-tested and often used transformer)
    * Added `TweetStim`, that can be initialized using Twitter's API, to the stimuli hierarchy
    * A update checker utility (in `pliers.utils`) that allows users to track changes to the output of different transformers for user-provided stimuli or a hand-selected battery of stimuli from the pliers test module
    * A revamped and improved `config` module with exposed accessor methods 
* New Transformers:
    * NLP Transformers:
        * TextVectorizerExtractor uses sklearn Vectorizers to extract bag-of-X features from a batch of text documents
        * WordEmbeddingExtractor uses gensim to extract word embeddings using a user-provided word2vec file
        * VADERSentimentExtractor uses nltk's VADER sentiment analysis tool to do sentiment analysis (w/o needing an API)
        * TokenizingFilter that splits a TextStim into several TextStims with one token in each
        * TokenRemovalFilter that removes specified tokens (e.g. stopwords, punctuation) from a TextStim
    * Audio feature extraction using `librosa`, wrapping all of the feature extraction methods `https://librosa.github.io/librosa/feature.html`
    * Image Filters
        * ImageCroppingFilter crops an image using a user-specified box (or just trims off black borders by default)
        * PillowImageFilter uses common simple filters from PIL to filter an image. Examples: edge, blur, contour, min filters
    * Google Vision API image web entity extraction
    * Face recognition extractor using the `face_recognition` package
* Fixes or enhancements to existing features:
    * Better identifying of stimulus uniqueness (for hashing purposes), to prevent strange merges when using `merge_results`
    * Temporary file management for transformers that require on-disk file names
    * Better support for exporting to vertically long Pandas dataframes
    * Added ability to flatten the multi-index returned from `merge_results`
    * Using v2 of the Clarifai API
    * Added ability to control whether or not metadata is presented in merged or single `ExtractorResult`s
    * Removed `DerivedVideoStim` class and replaced with `VideoFrameCollectionStim` to clean up `VideoStim` hierarchy
    * Standardized dependency handling with `attempt_to_import` function calls
    * Proper calculation of `AudioStim` `sampling_rate`
    * Added ability to use the `GoogleSpeechAPIConverter` to extract word onsets and durations as well
* A number of other minor bug fixes and cleaning of the codebase

## 0.1.1 (March 9, 2017)
This is a minor release that adds a number of new (minor) features and a bunch of bug fixes. Changes include:
* Added very basic parallelization support
* Added support for generating graph diagrams using PyGraphViz (via Graph.plot())
* Added a new RemoteStim that represents remote, uninitialized resources
* Added a new WordStemmingFilter that stems text using nltk
* Added support for Google Cloud Vision's safe image detection
* IBMSpeechAPIConverter now takes an argument indicating whether to return words or phrases
* Added support for image-based Indico API models
* Add support for multiple results returned by IBM speech converter
* General clean-up, code styling, and docstring addition
* Bug fixes:
    - Update various Transformers to reflect changes to APIs
    - Fixed dataset import bug that prevented PyPI-installed 0.1 from importing properly
    - Update args to numpy functions to ensure numpy 1.12+ compatibility
    - Various other minor bug fixes

## 0.1 (January 17, 2017)
First major release. Nothing before this counts.