.. include:: _includes/_replacements.rst

Transformers
============

As the name suggests, a |Transformer| is a kind of object that transforms other objects. In pliers, every |Transformer| always takes a single |Stim| as its input, though it can return different outputs. The |Transformer| API in pliers is modeled loosely on the widely-used scikit-learn API; as such, what defines a |Transformer|, from a user's perspective, is that one can always call pass a |Stim| instance to |Transformer|'s .transform() method and expect to get another object as a result.

In practice, most users should never have any reason to directly instantiate the base |Transformer| class. We will almost invariably work with one of three different |Transformer| sub-classes: |Extractor|, |Converter|, and |Filter|. These classes are distinguished by the type of output that their respective :py:`.transform()` methods produce:

================= ===== ======
Transformer class Input Output
================= ===== ======
Extractor         AStim ExtractorResult
Converter         AStim BStim
Filter            AStim AStim
================= ===== ======

Here, AStim and BStim are different |Stim| subclasses. So an |Extractor| always returns an |ExtractorResult|, no matter what type of |Stim| it receives as input. A |Converter| and a |Filter| are distinguished by the fact that a |Converter| always returns a |Stim| of a different class than its input, while a |Filter| always returns a |Stim| of the same type as its input. This simple hierarchy turns out to be extremely powerful, as it enables us to operate in a natural, graph-like way over |Stim|\s, by filtering and converting them as needed before applying one or more Extractors to obtain extracted feature values.

Let's examine each of these |Transformer| types more carefully.

Extractors
----------
|Extractor|\s are the most important kind of |Transformer| in pliers, and in many cases, users will never have to touch any other kind of |Transformer| directly. Every |Extractor| implements a :py:`transform()` method that takes a |Stim| object as its first argument, and returns an object of class |ExtractorResult| (see below). For example:

::

	# Google Cloud Vision API face detection
	from pliers.extractors import GoogleVisionAPIFaceExtractor

	ext = GoogleVisionAPIExtractor()
	result = ext.transform('my_image.jpg')


List of Extractor classes
~~~~~~~~~~~~~~~~~~~~~~~~~

At present, pliers implements several dozen |Extractor| classes that span a wide variety of input modalities and types of extracted features. These include:

.. currentmodule:: pliers.extractors

**Audio feature extractors**

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


**Image feature extractors**

.. autosummary::
	:toctree: generated/
 	:template: _class.rst

  BrightnessExtractor
  ClarifaiAPIExtractor
  GoogleVisionAPIFaceExtractor
  GoogleVisionAPILabelExtractor
  GoogleVisionAPIPropertyExtractor
  GoogleVisionAPISafeSearchExtractor
  GoogleVisionAPIWebEntitiesExtractor
  IndicoAPIImageExtractor
  SaliencyExtractor
  SharpnessExtractor
  TensorFlowInceptionV3Extractor
  VibranceExtractor


**Text feature extractors**

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


**Video feature extractors**

.. autosummary::
	:toctree: generated/
 	:template: _class.rst

  FarnebackOpticalFlowExtractor

Note that, in practice, the number of features one can extract using the above classes is extremely large, because many of these Extractors return open-ended feature sets that are determined by the contents of the input |Stim| and/or the specified initialization arguments. For example, most of the image-labeling Extractors that rely on deep learning-based services (e.g., |GoogleVisionAPILabelExtractor| and |ClarifaiAPIExtractor|) will return feature information for any of the top N objects detected in the image. And the |PredefinedDictionaryExtractor| provides a standardized interface to a large number of online word lookup dictionaries (e.g., word norms for written frequency, age-of-acquisition, emotionality ratings, etc.).

The ExtractorResult class
~~~~~~~~~~~~~~~~~~~~~~~~~

Calling transform() on an instantiated |Extractor| returns an object of class |ExtractorResult|. This is a lightweight container that contains all of the extracted feature information returned by the |Extractor|, and also stores references to the |Stim| and |Extractor| objects used to generate the result. The raw extracted feature values are stored in the .data property, but typically, we'll want to work with the data in a more convenient format. Fortunately, every |ExtractorResult| instance exposes a .to_df() methods that gives us a nice pandas DataFrame. You can refer back to our first Quickstart example to see this in action.

Merging Extractor results
~~~~~~~~~~~~~~~~~~~~~~~~~
In most cases, we'll want to do more than just apply a single |Extractor| to a single |Stim|. We might want to apply an |Extractor| to a set of stims (e.g., to run the GoogleVisionAPIFaceExtractor on a whole bunch of images), or to apply several different Extractors to a single |Stim| (e.g., to run both face recognition and object recognition services on each image). As we'll see later (in the section on |Graph|\s), pliers makes it easy to apply many Extractors to many |Stim|\s, and in such cases, it will automatically merge the extracted feature data into one big pandas DataFrame. But in cases where we're working with multiple results manually, we can still merge the results ourselves, using the appropriately named merge_results function.

Suppose we have a list of images, and we want to run both face recognition and object labeling on each image. Then we can do the following:

::

	from pliers.extractors import (GoogleVisionAPIFaceExtractor, GoogleVisionAPILabelExtractor)
	my_images = ['file1.jpg', 'file2.jpg', ...]

	face_ext = GoogleVisionAPIFaceExtractor()
	face_feats = face_ext.transform(my_images)

	lab_ext = GoogleVisionAPILabelExtractor()
	lab_feats = lab_ext.transform(my_images)

Now each of face_feats and lab_feats is a list of |ExtractorResult| objects. We could explicitly convert each element in each list to a pandas DataFrame (by calling .to_df()), but that would still be pretty unwieldy, as we would still need to figure out how to merge those DataFrames in a sensible way. Fortunately, merge_results will do all the work for us:

::

	from pliers.extractors import merge_results
	# merge_results expects a single list, so we concatenate our two lists
	df = merge_results(face_feats + lab_feats)

In the resulting DataFrame, every |Stim| is represented in a different row, and every feature is represented in a separate column. As noted earlier, the resulting DataFrame contains much more information than what's returned when we call to_df() on a single |ExtractorResult| object. Extra columns injected into the merged result include the name, class, filename (if any) and transformation history of each |Stim|; the name of each feature returned by each |Extractor|; and the name of each |Extractor| (as a second level in the column MultiIndex). You can prevent some of this additional information from being added by setting the :py:`extractor_names` and :py:`stim_names` arguments in :py:func:`merge_results` to :py:`False` (by default, both are :py:`True`).

Converters
----------
Converters, as their name suggests, convert |Stim| classes from one type to another. For example, the |IBMSpeechAPIConverter|, which is a subclass of |AudioToTextConverter|, takes an |AudioStim| as input, queries IBM's Watson speech-to-text API, and returns a transcription of the audio as a |ComplexTextStim| object. Most |Converter| classes have sensible names that clearly indicate what they do, but to prevent any ambiguity (and support type-checking), every concrete |Converter| class must define :py:`_input_type` and :py:`_output_type` properties that indicate what |Stim| classes they take and return as input and output, respectively.

Implicit Stim conversion
~~~~~~~~~~~~~~~~~~~~~~~~
Although Converters play a critical role in pliers, they usually don't need to be invoked explicitly by users, as pliers can usually figure out what conversions must be performed and carry them out implicitly. For example, suppose we want to run the |STFTAudioExtractor|---which computes the short-time Fourier transform on an audio clip and returns its power spectrum---on the audio track of a movie clip. We don't need to explicitly convert the |VideoStim| to an |AudioStim|, because pliers is clever enough to determine that it can get the appropriate input for the |STFTAudioExtractor| by executing the |VideoToAudioConverter|. In practice, then, the following two snippets produce identical results:

::

	from pliers.extractors import STFTAudioExtractor
	from pliers.stimuli import VideoStim
	video = VideoStim('my_movie.mp4')

	# Option A: explicit conversion
	from pliers.converters import VideoToAudioConverter
	conv = VideoToAudioConverter()
	audio = conv.transform(video)
	ext = STFTAudioExtractor(freq_bins=10)
	result = ext.transform(audio)

	# Option B: implicit conversion
	ext = STFTAudioExtractor(freq_bins=10)
	result = ext.transform(video)

Because pliers contains a number of "multistep" |Converter| classes, which chain together multiple standard Converters, implicit |Stim| conversion will typically work not only for a single conversion, but also for a whole series of them. For example, if you feed a video file to a |LengthExtractor| (which just counts the number of characters in each TextStim's text), pliers will use the built-in VideoToTextConverter class to transform your |VideoStim| into a |TextStim|, and everything should work smoothly in most cases.

I say "most" cases, because there are two important gotchas to be aware of when relying on implicit conversion. First, sometimes there's an inherent ambiguity about what trajectory a given stimulus should take through converter space; in such cases, the default conversions pliers performs may not line up with your expectations. For example, a |VideoStim| can be converted to a |TextStim| either by (a) extracting the audio track from the video and then transcribing into text via a speech recognition service, or (b) extracting the video frames from the video and then attempting to detect any text labels within each image. Because pliers has no way of knowing which of these you're trying to accomplish, it will default to the first. The upshot is that if you think there's any chance of ambiguity in the conversion process, it's probably a good idea to explicitly chain the |Converter| steps (you can do this very easily using the |Graph| interface discussed below). The explicit approach also provides additional precision in that you may want to initialize a particular |Converter| with non-default arguments, and/or specify exactly which of several candidate |Converter| classes to use (e.g., pliers defaults to performing speech-to-text conversion via the IBM Watson API, but also provides alternative support for the Wit.AI, and Google Cloud Speech APIs services).

.. _conversion-defaults:

Package-wide conversion defaults
################################

Alternatively, you can set the default Converter(s) to use for any implicit |Stim| conversion at a package-wide level, via the config.default_converters attribute. By default, this is something like:

::

	default_converters = {
	    'AudioStim->TextStim': ('IBMSpeechAPIConverter', 'WitTranscriptionConverter'),
	    'ImageStim->TextStim': ('GoogleVisionAPITextConverter', 'TesseractConverter')
	}

Here, each entry in the default_converters dictionary lists the Converter(s) to use, in order of preference. For example, the above indicates that any conversion between |ImageStim| and |TextStim| should first try to use the |GoogleVisionAPITextConverter|, and then, if that fails (e.g., because the user has no Google Cloud Vision API key set up), fall back on the |TesseractConverter|. If all selections specified in the config fail, pliers will still try to use any matching Converters it finds, but you'll lose the ability to control the order of selection.

Second, because many Converters call API-based services, if you're going to rely on implicit conversion, you should make sure that any API keys you might need are properly set up as environment variables in your local environment, seeing as you're not going to be able to pass those keys to the Converter as initialization arguments. For example, by default, pliers uses the IBM Watson API for speech-to-text conversion (i.e., when converting an |AudioStim| to a |ComplexTextStim|). But since you won't necessarily know this ahead of time, you won't be able to initialize the Converter with the correct credentials--i.e., by calling :py:`IBMSpeechAPIConverter(username='my_username', password='my_password')`. Instead, the Converter will get initialized without any arguments (:py:`IBMSpeechAPIConverter()`), which means the initialization logic will immediately proceed to look for IBM_USERNAME and IBM_PASSWORD variables in the environment, and will raise an exception if at least one of these variables is missing. So make sure as many API keys as possible are appropriately set in the environment. You can read more about this in the API keys section.

List of Converter classes
~~~~~~~~~~~~~~~~~~~~~~~~~
Pliers currently implements the following |Converter| classes:

.. currentmodule:: pliers.converters

.. autosummary::
	:toctree: generated/
  :template: _class.rst

	ComplexTextIterator
	IBMSpeechAPIConverter
	GoogleSpeechAPIConverter
	GoogleVisionAPITextConverter
	TesseractConverter
	VideoFrameCollectionIterator
	VideoFrameIterator
	VideoToAudioConverter
	VideoToComplexTextConverter
	VideoToTextConverter
	WitTranscriptionConverter

Filters
-------
A :py:`Filter` is a kind of |Transformer| that returns an object of the same |Stim| class as its input. Filters can be used for tasks like image or audio filtering, text tokenization or sanitization, and many other things. The defining feature of a |Filter| class is simply that it must return a |Stim| of the same type as the input passed to the :py:`.transform()` method (e.g., passing in an |ImageStim| and getting back another, modified, |ImageStim|).

List of Filter classes
~~~~~~~~~~~~~~~~~~~~~~~~~
Pliers currently implements the following |Filter| classes:

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

Iterable-aware transformations
------------------------------
A useful feature of the |Transformer| API is that it's inherently iterable-aware: every pliers |Transformer| (including all Extractors, Converters, and Filters) can be passed an iterable (specifically, a list, tuple, or generator) of |Stim| objects rather than just a single |Stim|. The transformation will then be applied independently to each |Stim|.
