.. include:: _includes/_replacements.rst

Basic concepts
==============

Pliers is designed to provide a standardized, intuitive interface to a wide range of feature extraction tools and services. As such, the API is deliberately designed with simplicity in mind. It's modeled loosely on patterns used in `scikit-learn <http://scikit-learn.org/>`_ (and now adopted by many other Python packages for machine learning and data science).

Stims and Transformers
----------------------

At its core, the pliers architecture is based around two kinds of objects (and the interactions between them): the |Stim|, and the |Transformer|. A |Stim| class ('stim' is short for stimulus) is a pliers container for an object we want to extract features from --for example, a video clip or a text excerpt. A |Transformer| class provides functionality to *do* something with Stims--either to change the input Stim in some way (e.g.,by modifying its data, or turning it into a different kind of |Stim|), or to extract feature data from it.

To illustrate, consider a very simple example:

::

	from pliers.stimuli import ImageStim
	from pliers.extractors import GoogleVisionAPIFaceExtractor

	stim = ImageStim('my_image.jpg')
	ext = GoogleVisionAPIFaceExtractor()
	face_features = ext.transform(stim)

The separation between the target of feature extraction (a JPEG image that we're storing as an |ImageStim|) and the |Extractor| object that *transforms* that target is made clear here. This basic pattern persists throughout pliers workflows, no matter how many Transformers we string together, or how many Stim inputs we feed in.

Types of Transformers
---------------------
Virtually all of the objects that perform operations on Stims in pliers are |Transformer|\s. However, there are three distinct types of |Transformer| classes that users should be aware of:

1. **Extractors** are Transformers that take a |Stim| as input and return some extracted feature data in the form of a |ExtractorResult| object (or, optionally, a pandas DataFrame). The |ExtractorResult| represents the natural end point of a pliers workflow. ExtractorResults contain extracted feature data (e.g., in the example above, the :py:`face_features` variable would contain information about any faces detected in the input image).

2. **Converters** are Transformers that take a |Stim| object as input, and convert it to a different type of |Stim|. For example, the |VideoToAudioConverter| takes a |VideoStim| as input and returns an |AudioStim| as output (it does this simply by extracting the audio track from the video and wrapping it in a new |AudioStim| instance).

3. **Filters** are Transformers that take a |Stim| object as input, and return a modified |Stim| of the *same* class. For example, the |ImageCroppingFilter| takes an |ImageStim| as input, and returns another |ImageStim| as output, where the image data stored in the Stim has been cropped in accordance with a bounding box specified when the |ImageCroppingFilter| was initialized.

In practice, most users will interact frequently with Extractors, occasionally with Converters, and very rarely if ever with Filters. Part of the reason for this is that, as we'll see in the next section, most conversion between |Stim| types can be accomplished implicitly in pliers, which can greatly reduce the amount of code one has to write in order to construct powerful feature extraction workflows (though at the potential cost of clarity).

For further details on the different |Transformer| types--as well as lists of all available classes--see the :doc:`transformers` section.

Chaining transformers
---------------------

The Transformers implemented in pliers can individually be quite powerful--as exemplfied by the simple face detection example above. But the real power of the package comes into focus once we start combining Transformers. By connecting |Converter|, |Filter|, and |Extractor| classes together into simple graphs (technically, trees), we open the door to extremely flexible, yet surprisingly terse, feature extraction workflows.

Here's a minimal example that builds on our previous example using all three types of Transformers, and illustrates how easy it is to effectively "chain" Transformers:

::

	from pliers.stimuli import VideoStim
	from pliers.converters import VideoFrameIterator
	from pliers.filters import FrameSamplingFilter
	from pliers.extractors import GoogleVisionAPIFaceExtractor, merge_results
 
	video = VideoStim('my_movie.mpg')

	# Convert the VideoStim to a list of ImageStims
	conv = VideoFrameIterator()
	frames = conv.transform(video)

	# Sample 2 frames per second
	filt = FrameSamplingFilter(hertz=2)
	selected_frames = filt.transform(frames)

	# Detect faces in all frames
	ext = GoogleVisionAPIFaceExtractor()
	face_features = ext.transform(selected_frames)

	# Merge results from all frames into one big pandas DataFrame
	data = merge_results(face_features)

We begin by reading in a video as a single |VideoStim|. We then explicitly convert the video to a set of discrete |ImageStim| objects (which allows us to use |Extractor| classes that expect image inputs). Next, we filter the list of video frames to retain only 2 frames per second (perhaps because we want to save ourselves some money by sending google fewer images to annotate). Lastly, we use the |GoogleVisionAPIFaceExtractor| to run face detection on each of the remaining images, just as in our previous example.

Next steps
----------

The above overview should give you a basic sense of the pliers architecture and interface, but there are some other points worth being aware of (all of which are described in more detail elsewhere in the docs):

1. *Implicit Stim conversion*. Many conversion steps can be performed implicitly; e.g., in the listing above, we could have left out the |VideoFrameIterator| step entirely, and the |FrameSamplingFilter| would have performed the conversion implicitly. In fact, if we didn't need to downsample the video to just 2 frames/second, we could also have left out the |FrameSamplingFilter| step, and the |GoogleVisionAPIFaceExtractor| would have then automatically performed face detection on every single frame of video. 

2. *The Graph abstraction*. While it's generally good to be explicit, in cases where one needs to chain together a large number of Transformers, an explicit can make it quite hard to understand what's going on at a glance. Fortunately, pliers provides a :doc:`graph <graphs>` abstraction that makes it much easier to specify complex graphs. It also provides rudimentary plotting capabilities, making it even easier to understand what's happening.

3. *Package-level configuration*. Pliers has a number of package-level settings that can be used to change the package's behavior in important ways. These include basic control over caching, logging, and automatic parallelization, among other things. For more information, see the :doc:`config` section.
