# Changelog

## 0.4.1 (February 26, 2021)
This is a minor release that adds one new extractor, and a resampling utility, as well as a large number of minor bugfixes.

New features:
* Add TensorFlow hub extractor (#433)
* Add ExtractedFeature resample utility (#430)

Bug fixes and other house keeping
* Replace BERT with lighter extract in metric extractor tests 
* Fix pytest memory leak by running forked (#439)
* Remove depracated pandas methods (#438)
* Convert CI to Github Actions (#436)
* Update HuggingFace BERT Extractors (#435)
* Resample upsampling and add upsampling test case (#431)
* Librosa links (#429)
* Google API Regressions (#424)
* Access string in numpy array (#422)
* Update REAME to link to correct pliers docs page (#421)
* Test regressions (#418)
* Remove --no-user-group (#414)
* Use multi-stage builds in Dockerfile to reduce final image size (#411)
* Remove instance of Python2 super (#410)
* Update librosa feature links (#409)
* If Google Face Extractor result landmark has no type, skip. (#406)
* librosa import error (numba) (#405)
* Remove soundfile as dependency (#408)

## 0.4.0 (July 14, 2020)
This is a major release that adds support for several new feature extraction services/models, as well as a large number of minor bugfixes and improvements.

New features and major improvements/bugfixes:
* Add WordCounterExtractor (#364)
* Add PretrainedBertEncodingExtractor from transformers #365
* Add lemmatization to stemming filter (WordStemmingFilter) #366
* Add Lancaster Sensorimotor Norms to PredefinedDictionaryExtractor #367
* Remove deprecated Indico API extractors #369
* Add AudioResamplingFilter #374
* Add AudiosetLabelExtractor from Yamnet architecture #379 #399
* Add BertLMExtractor, BertSentimentExtractor, BertSequenceEncodingExtractor #383 #397
* Add log attributes to to_df #389
* Add MetricExtractor #390 #398
* Add ExtractorResult --> SeriesStim converter #394

Minor improvements and bug fixes:
* Fixes case of missing landmark type in GoogleVisionAPIFaceExtractor (#406)
* Removed soundfile as dependency (#408)
* Fix librosa import erroor (#405)
* Fixes to Clarifai face extractor (#335, #357)
* Added optional dependencies as 'extras' for pip install #358
* Patch operations request for Google Video Intelligence #360
* Miscellaneous fixes to tests and Travis CI #372 #375 #378 #380 #381
* Fix index_cols behavior in to_df function  #376 #388
* Drop vestigial Python 2 code #392
* Upgrade Python syntax and add python_requires for 3.5+ #393

## 0.3.0 (June 8, 2019)
This is a major release that adds support for several new feature extraction services/models, as well as a large number of minor bugfixes and improvements.

New features and major improvements/bugfixes:
* Basic support for Google Natural Language API (#306)
* Basic support for Google Video Intelligence API (#288)
* Support for Rev.ai speech-to-text converter (#334)
* Added several librosa features (#320)
* Support for pre-trained Keras-based image classifiers (#341, #344)
* Added .srt output functionality to `ComplexTextStim` (#301)
* Added a `get_bytestring()` method to `ImageStim` and `VideoStim` classes (#324)
* Expanded/improved `PredefinedDictionaryExtractor` list (#339, #351)
* Adds `ImageResizeFilter` (#342)
* Improved/fixed Dockerfile (#343)

Minor improvements and bug fixes:
* Fixed bug in `GoogleSpeechAPIConverter` that overwrote all but last speech block (#316)
* Improved sampling accuracy in FrameSamplingFilter (#301)
* Fixed bug in RGB information returned by GoogleVisionAPI (#296)
* Fixed caching of transformations on file paths (#286)
* Adds support for a few config options in Google Video Intelligence extractors (#337)
* Various other minor improvements and fixes (#301, #308, #312, #324, #332)

## 0.2.3 (April 7, 2018)
This is a minor release that adds several new features and bug fixes:
* A number improvements to API transformers (custom marker for unit tests; key validation; large job limits; etc.; see #270)
* Enhance API transformers to work from remote URLs if available, without requiring local file download (#282)
* Protected timing variable names to prevent collisions in `to_df` calls (#281)
* Add caching to transformers that inherit from `BatchTransformerMixin` (#283)
* Fixes to update utility (#273)
* Updated docstrings (#282)

## 0.2.2 (March 1, 2018)
This is a minor release that adds several new features and bug fixes:
* Support for several Microsoft Vision services, including the Face API and Vision API (#259)
* Improvements to the Graph API (#254):
    * Graphs can now return Stims in cases where terminal nodes are Converters or Filters
    * Serialization and de-serialization from JSON
    * A helper `add_chain` method that simplifies construction of completely linear Graphs
    * Improved graph plotting, including representation of node types using color and line type
* Improved internal handling and propagation of temporal properties (onset, duration, and order; #261)
* Refactored the internal handling/formatting of raw feature extraction results (#261)
* Added temporal cropping filters that make it easy to crop audio and video clips to specified boundaries (#244)
* Reorganization of the API Transformer hierarchy (#266)
* Minor improvements to to text filters (e.g., lower-casing and tokenization; #246)
* Added basic support for duecredit (thanks to @yarikoptic; #254)
* Several minor bug fixes

## 0.2.1 (January 31, 2018)
This is a bugfix release that addresses an installation bug on bare environments (thanks to @mgxd).

## 0.2 (January 29, 2018)
This is a major release that adds many new features and transformers. Changes include:
* Sphinx [documentation](http://tyarkoni.github.io/pliers/)!
* New package features
    * An updated extractor result management API with simplified merging mechanisms (see #232 for more details)
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
