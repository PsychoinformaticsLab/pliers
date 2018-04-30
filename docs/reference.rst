--------------------
Pliers API Reference
--------------------

This is the full API reference for all user-facing classes and functions in the pliers package.

Converters (:mod:`pliers.converters`)
-------------------------------------

.. automodule:: pliers.converters
	:no-members:
	:no-inherited-members:

**Classes**:

.. currentmodule:: pliers.converters

.. autosummary::
	:toctree: generated/
  :template: _class.rst

	ComplexTextIterator
	IBMSpeechAPIConverter
	GoogleSpeechAPIConverter
	GoogleVisionAPITextConverter
  MicrosoftAPITextConverter
	TesseractConverter
	VideoFrameCollectionIterator
	VideoFrameIterator
	VideoToAudioConverter
	VideoToComplexTextConverter
	VideoToTextConverter
  WitTranscriptionConverter

**Functions**:

.. currentmodule:: pliers.converters

.. autosummary::
  :toctree: generated/
  :template: _function.rst

  get_converter


Dataset utilities (:mod:`pliers.datasets`)
------------------------------------------

.. automodule:: pliers.datasets
	:no-members:
	:no-inherited-members:

**Functions**:

.. currentmodule:: pliers.datasets

.. autosummary::
  :toctree: generated/
  :template: _function.rst

  fetch_dictionary


Diagnostic utilities (:mod:`pliers.diagnostics`)
------------------------------------------------

.. automodule:: pliers.diagnostics
	:no-members:
	:no-inherited-members:

**Classes**:

.. currentmodule:: pliers.diagnostics

.. autosummary::
	:toctree: generated/
  :template: _class.rst

  Diagnostics

**Functions**:

.. currentmodule:: pliers.diagnostics

.. autosummary::
   :toctree: generated/
   :template: _function.rst

   correlation_matrix
   eigenvalues
   condition_indices
   variance_inflation_factors
   mahalanobis_distances
   variances


Extractors (:mod:`pliers.extractors`)
-------------------------------------

.. automodule:: pliers.extractors
	:no-members:
	:no-inherited-members:

.. currentmodule:: pliers.extractors

**Classes**:

*Base extractors and associated objects*

.. autosummary::
	:toctree: generated/
	:template: _class.rst

	Extractor
	ExtractorResult


*Audio feature extractors*

.. autosummary::
	:toctree: generated/
 	:template: _class.rst

  ChromaCENSExtractor
  ChromaCQTExtractor
  ChromaSTFTExtractor
  MeanAmplitudeExtractor
  MelspectrogramExtractor
  MFCCExtractor
  PolyFeaturesExtractor
  RMSEExtractor
  SpectralCentroidExtractor
  SpectralBandwidthExtractor
  SpectralContrastExtractor
  SpectralRolloffExtractor
  STFTAudioExtractor
  TempogramExtractor
  TonnetzExtractor
  ZeroCrossingRateExtractor


*Image feature extractors*

.. autosummary::
	:toctree: generated/
 	:template: _class.rst

  BrightnessExtractor
  ClarifaiAPIImageExtractor
  ClarifaiAPIVideoExtractor
  FaceRecognitionFaceEncodingsExtractor
  FaceRecognitionFaceLandmarksExtractor
  FaceRecognitionFaceLocationsExtractor
  GoogleVisionAPIFaceExtractor
  GoogleVisionAPILabelExtractor
  GoogleVisionAPIPropertyExtractor
  GoogleVisionAPISafeSearchExtractor
  GoogleVisionAPIWebEntitiesExtractor
  IndicoAPIImageExtractor
  MicrosoftAPIFaceExtractor
  MicrosoftAPIFaceEmotionExtractor
  MicrosoftVisionAPIExtractor
  MicrosoftVisionAPITagExtractor
  MicrosoftVisionAPICategoryExtractor
  MicrosoftVisionAPIImageTypeExtractor
  MicrosoftVisionAPIColorExtractor
  MicrosoftVisionAPIAdultExtractor
  SaliencyExtractor
  SharpnessExtractor
  TensorFlowInceptionV3Extractor
  VibranceExtractor


*Text feature extractors*

.. autosummary::
	:toctree: generated/
 	:template: _class.rst

  ComplexTextExtractor
  DictionaryExtractor
  IndicoAPITextExtractor
  LengthExtractor
  NumUniqueWordsExtractor
  PartOfSpeechExtractor
  PredefinedDictionaryExtractor
  TextVectorizerExtractor
  VADERSentimentExtractor
  WordEmbeddingExtractor


*Video feature extractors*

.. autosummary::
	:toctree: generated/
 	:template: _class.rst

  FarnebackOpticalFlowExtractor


**Functions**:

.. currentmodule:: pliers.extractors

.. autosummary::
  :toctree: generated/
  :template: _function.rst

  merge_results


Filters (:mod:`pliers.filters`)
-------------------------------

.. automodule:: pliers.filters
	:no-members:
	:no-inherited-members:

**Classes**:

.. currentmodule:: pliers.filters

.. autosummary::
	:toctree: generated/
 	:template: _class.rst

  FrameSamplingFilter
  ImageCroppingFilter
  PillowImageFilter
  PunctuationRemovalFilter
  TokenizingFilter
  TokenRemovalFilter
  WordStemmingFilter


Graph construction (:mod:`pliers.graph`)
----------------------------------------

.. automodule:: pliers.graph
	:no-members:
	:no-inherited-members:

**Classes**:

.. currentmodule:: pliers.graph

.. autosummary::
	:toctree: generated/
 	:template: _class.rst

	Graph
	Node


Stimuli (:mod:`pliers.stimuli`)
-------------------------------

.. automodule:: pliers.stimuli
	:no-members:
	:no-inherited-members:

**Classes**:

.. currentmodule:: pliers.stimuli

.. autosummary::
	:toctree: generated/
 	:template: _class.rst

  AudioStim
  ComplexTextStim
  CompoundStim
  ImageStim
  TextStim
  TweetStimFactory
  TweetStim
  TranscribedAudioCompoundStim
  VideoFrameCollectionStim
  VideoFrameStim
  VideoStim

**Functions**:

.. currentmodule:: pliers.stimuli

.. autosummary::
   :toctree: generated/
   :template: _function.rst

    load_stims


Transformers (:mod:`pliers.transformers`)
-----------------------------------------

.. automodule:: pliers.transformers
	:no-members:
	:no-inherited-members:

**Classes**:

.. currentmodule:: pliers.transformers

.. autosummary::
	:toctree: generated/
 	:template: _class.rst

	BatchTransformerMixin
	GoogleAPITransformer
	GoogleVisionAPITransformer
	Transformer

**Functions**:

.. currentmodule:: pliers.transformers

.. autosummary::
   :toctree: generated/
   :template: _function.rst

   get_transformer
