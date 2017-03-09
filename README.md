# pliers

A Python package for automated extraction of features from multimodal stimuli.

## Status
* [![Build Status](https://travis-ci.org/tyarkoni/pliers.svg?branch=master)](https://travis-ci.org/tyarkoni/pliers)
* [![Coverage Status](https://coveralls.io/repos/github/tyarkoni/pliers/badge.svg?branch=master)](https://coveralls.io/github/tyarkoni/pliers?branch=master)

## Overview

Pliers is a Python package for automated extraction of features from multimodal stimuli. It's designed to let you rapidly and flexibly extract all kinds of useful information from videos, images, audio, and text. It's also easily extensible, allowing users to write new feature extractors in relatively little code, and providing a common framework for interfacing with all kinds of domain-specific tools.

Pliers is still in early development, so the API may occasionally break, though no major changes are rare at this point.

## Installation

For the latest release:

> pip install pliers

Or, if you want to work on the bleeding edge:

> pip install pliers git+https://github.com/tyarkoni/pliers.git

## Dependencies

All mandatory Python dependencies (listed in [requirements.txt](https://github.com/tyarkoni/pliers/blob/master/requirements.txt)) should be automatically installed when pliers is installed. Additionally, there are a number of optional dependencies that you may want to install depending on what kinds of features you plan to extract (see [optional-dependencies.txt](https://github.com/tyarkoni/pliers/blob/master/optional-dependencies.txt)). To be on the safe side, you can install all of the optional dependencies with pip:

> pip install -r optional-dependencies.txt

**Note**: some of pliers's Python dependencies have their own (possibly platform-dependent) requirements. Most notably, [python-magic](https://github.com/ahupp/python-magic) requires libmagic (see [here](https://github.com/ahupp/python-magic#dependencies) for installation instructions), and without this, you'll be relegated to loading all your stims explicitly rather than passing in filenames (i.e., `stim = VideoStim('my_video.mp4')` will work fine, but passing `'my_video.mp4'` directly to an `Extractor` will not). Additionally, the Python OpenCV bindings unsurprisingly require [OpenCV3](http://opencv.org/) (which can be a bit more challenging to [install](http://docs.opencv.org/2.4/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html)), but relatively few of the feature extractors in pliers currently depend on OpenCV, so you may not need to bother with this.

## Table of Contents
- [Quickstart](#quickstart)
    + [Example the First](#example-the-first)
    + [Example the Second](#example-the-second)
- [User Guide](#user-guide)
    + [Stims](#stims)
        * [Iterable Stims](#iterable-stims)
        * [Temporal properties](#temporal-properties)
        * [Supported Stim types](#supported-stim-types)
        * [Intelligent Stim loading](#intelligent-stim-loading)
        * [Transformation history](#transformation-history)
    + [Transformers](#transformers)
        * [Extractors](#extractors)
            - [Available Extractors](#available-extractors)
            - [ExtractorResults](#extractorresults)
        * [Converters](#converters)
            - [Implicit Stim conversion](#implicit-stim-conversion)
            - [Available Converters](#available-converters)
        * [Filters](#filters)
            - [Available Filters](#available-converters)
        * [Iterable-aware transformations](#iterable-aware-transformations)
    + [API keys](#api-keys)
    + [Graphs](#graphs)
    + [Configuration](#configuration)
    + [Common gotchas](#common-gotchas)
    + [Bug reports etc.](#bug-reports)
- [Developer Guide](#developer-guide)
    + [General overview](#general-overview)
    + [Writing new Transformers](#writing-new-transformers)
    + [Conventions](#conventions)
    + [Testing](#testing)

## Quickstart

The fastest way to learn how pliers works, and appreciate its power and flexibility, is to work through a couple of short examples.

### Example the First
This first example demonstrates how to run a single image through a single "Extractor" and return some features in an accessible form. We'll use Google's Cloud Vision Face Extraction API, which returns all kinds of information about faces--e.g., where in an image a face occurs; whether the person in the image is wearing a hat; and what facial expressions (if any) are detected.

The totality of the code required to do this (assuming you have a [Google Cloud Vision API key](#api-keys) set up, which we'll discuss later) is the following:

```python
from pliers.extractors import GoogleVisionAPIFaceExtractor

ext = GoogleVisionAPIFaceExtractor()
result = ext.transform('obama.jpg').to_df()
```

Suppose our input looks like this:

![Obama](https://raw.githubusercontent.com/tyarkoni/pliers/master/pliers/tests/data/image/obama.jpg)

Then the result returned by the Extractor, which we've conveniently represented as a pandas DataFrame, look like this:

    angerLikelihood                      VERY_UNLIKELY
    blurredLikelihood                    VERY_UNLIKELY
    boundingPoly_vertex1_x                          34
    boundingPoly_vertex1_y                           3
    boundingPoly_vertex2_x                         413
    boundingPoly_vertex2_y                           3
    boundingPoly_vertex3_x                         413
    boundingPoly_vertex3_y                         444
    boundingPoly_vertex4_x                          34
    boundingPoly_vertex4_y                         444
    duration                                      None
    face_detectionConfidence                  0.999946
    ...

As you can see, pliers is (courtesy of Google) very confident that Barack Obama has a face.

### Example the Second
The above example illustrates the application of a single extractor to a single input image. But pliers allows us to construct entire graphs that handle multiple inputs passed through multiple Extractors, Converters, and Filters (a more detailed explanation of each of these things can be found [below](#transformers)). In this example, we construct a `Graph` that takes in a list of video clips (`VideoStim`) as input and passes each one through a series of transformations. For each video, we extract:

* Image labels for image frames extracted from the movie at regular intervals, using deep learning models from two different sources (a local pre-trained deep convolutional neural net model, and the [Clarifai](http://clarifai.com) web API);
* Power in different parts of the audio track's frequency spectrum;
* A transcription of speech detected in the movie; and
* Word-by-word sentiment analysis and frequency norms for the extracted transcripts.

Note that extracting these features requires a series of implicit transformations of the original inputs: for each video, we need to extract the audio track, and then submit the extracted audio to a speech transcription API to obtain text. Fortunately, pliers does almost all this for us implicitly; we don't need to explicitly convert between `Stim` types ourselves at almost any step (actually, technically, it could do *all* of this for us implicitly, but for image labeling, we probably don't want to process every single movie frame, since on-screen information doesn't change that fast, so we'll slow down the sampling rate via an explicit frame-sampling step).

```python
from pliers.stimuli import VideoStim
from pliers.extractors import (STFTAudioExtractor, PredefinedDictionaryExtractor, ComplexTextExtractor, ClarifaiAPIExtractor, IndicoAPIExtractor)
from pliers.converters import FrameSamplingConverter
from pliers.graph import Graph

# Initialize the video clips from files as new VideoStims. Note that the
# second line is optional; pliers will infer that the clips are video
# files based on their extensions. But it's good practice to be explicit.
clips = ['../pliers/tests/data/video/obama_speech.mp4']
clips = [VideoStim(f) for f in clips]

# Initialize graph--note that we don't need to include any Stim conversion
# nodes, as they will be injected automatically. We make an exception for
# the FrameSamplingConverter, because we don't want to analyze every
# single video frame--that would just waste resources. Also note that each
# element in the node list can be either a string or an Extractor instance.
# We don't need to bother initializing Extractors unless we need to pass
# arguments.
nodes = [
    (FrameSamplingConverter(hertz=1),
         ['ClarifaiAPIExtractor', 'TensorFlowInceptionV3Extractor']),
    STFTAudioExtractor(hop_size=1, freq_bins=[(100, 300), (300, 3000), (3000, 20000)]),
    PredefinedDictionaryExtractor(['SUBTLEXusfrequencyabove1/Lg10WF']),
    IndicoAPIExtractor(models=['sentiment'])
]
g = Graph(nodes)

# Execute graph and collect results
result = g.run(clips)
```
The above listing is very terse, and you can read more about the [Graph API](#graphs) to understand what's going on here. But the bottom line is that, in just a handful of lines of code, we've managed to extract a number of very different features spanning several modalities and pulled from several sources, and unite them into a common representation.

By default, `Graph.extract()` collects all features from all Extractors and merges them into one big pandas DataFrame in "tidy" format, where the columns are features and each row is a single event (i.e., a segment of the original stimulus with a specified onset and duration). In our case, if we look at one of the first few rows, we see something like this (partial listing):

| | | |
--------------------|--------------|--------
ClarifaiAPIExtractor|administration|0.971645
||adult|0.822863
||authority|NaN
||award|0.895437
||business|0.944746
||ceremony|NaN
||chair|0.954877
|...|...|...
IndicoAPIExtractor|sentiment|NaN
PredefinedDictionaryExtractor|SUBTLEXusfrequencyabove1_Lg10WF|NaN
STFTAudioExtractor|100_300|NaN
||3000_20000|NaN
||300_3000|NaN
TensorFlowInceptionV3Extractor|label_1|Windsor tie
||label_2|suit, suit of clothes
||score_1|0.52688
|...|...|...
class||VideoFrameStim
history||VideoStim->FrameSamplingConverter/DerivedVideo...
onset||0
stim||frame[0]

For clarity, a bunch of the values are omitted, but the idea should be clear. Note that many cells are missing values; the row we're seeing here reflects a single video frame, which, as an image, obviously cannot have values for the audio and text extractors. Conversely, other rows in the DataFrame will have values for these other extractors,but not for the image labeling extractors. While this probably isn't the cleanest, best-organized DataFrame you've ever seen, it does consolidate a lot of information in one place. In addition to just the feature data, we also get a bunch of extra columns (e.g., we get to see the full conversion history for each `Stim`; the column names reflect the source `Extractor`s; and an `onset` column telling us that the frame was presented at time `0` relative to the start of the video). You can always ignore the stuff you don't care about--the pliers philosophy is to err on the side of providing more rather than less information.

## User Guide

### Stims
The core data object used in pliers is something called a `Stim` (short for stimulus). A `Stim` instance is a lightweight wrapper around any one of several standard file or data types: a video file, an image, an audio file, or some text. Under the hood, pliers uses other Python libraries to support most operations on the original files (e.g., [MoviePy](https://github.com/Zulko/moviepy/) for video and audio files, and [pillow/PIL](https://python-pillow.org/) for images).

We can minimally initialize new `Stim` instances by passing in a filename to the appropriate class initializer. Alternatively, some `Stim` types also accept a `data` argument that allows initialization from an appropriate data type instead of a file (e.g., a numpy array for images, a string for text, etc.). Examples:

```python
from pliers.stimuli import VideoStim, ImageStim, TextStim

# Initialize from files
vs = VideoStim('my_video_file.mp4')
img1 = ImageStim('first_image.jpg')

# Initialize a TextStim from a string
text = TextStim("this sentence will be represented as a single token of text")

# Initialize an ImageStim from a random RGB array
img_data = np.random.uniform(size=(100, 100, 3))
img2 = ImageStim(data=img_data)
```

In general, users will rarely have a need to directly manipulate `Stim` classes, beyond them passing them around to `Extractor`s. In fact, it's often not necessary to initialize `Stim`s at all, as you can generally pass a list of string filenames to any `Extractor` (with some caveats discussed [below](#intelligent-stim-loading).

All `Stim` classes expose a number of attributes that provide some information about the associated stimulus. Most notably, the `.filename` and `.name` attributes store the names of the source file (e.g., `my_movie.mp4`) and an (optional) derived name, respectively.

### Iterable Stims

Some `Stim` classes are naturally iterable, so pliers makes it easy to iterate their elements. For example, a `VideoStim` is made up of a series of `VideoFrameStims`, and a `ComplexTextStim` is made up of `TextStims`. Looping over the constituent elements is trivial:

```python
>>> from pliers.stimuli import ComplexTextStim
>>> cts = ComplexTextStim(text="This class method uses the default nltk word tokenizer, which will split this sentence into a list of TextStims, each representing a single word.")
>>> for ts in cts:
>>>    print(ts.text)
"This"
"class"
"method"
...
```

You can also directly access the constituent elements within the containing `Stim` if you need to (e.g., `VideoStim.frames` or `ComplexTextStim.elements`). Note that, for efficiency reasons, these properties will typically return generators rather than lists (e.g., retrieving the `.frames` property of a `VideoStim` will return a generator that reads frames lazily). You can always explicitly convert the generator to a list (e.g., `frames = list(video.frames)`), just be aware that your memory footprint may instantly balloon in cases where you're working with large media files.

### Temporal properties
Some `Stim` classes inherently have a temporal dimension (e.g., `VideoStim` and `AudioStim`). However, even a static `Stim` instance such as an `ImageStim` or a `TextStim` will often be assigned `.onset` and `.duration` properties during initialization. Typically, this happens because the static `Stim` is understood to be embedded within some temporal context. Consider the following code:

```python
>>> vs = VideoStim('my_video.mp4')
>>> frame_200 = vs.get_frame(200)
>>> print(frame_200.onset, frame_200.duration)
6.666666666666667, 0.03333333333333333
```

Assume the above video runs at 30 frames per second. Then, when we retrieve the 200th video frame, the result is a `VideoFrameStim` that, in addition to storing a numpy array in its `.data` attribute (i.e., the actual image content of the frame), also keeps track of (a) the onset of the frame (in seconds) relative to the start of the source video file, and (b) the duration for which the `VideoFrameStim` is supposed to be presented (in this case 1/30 seconds).

Although `Stim` onsets and/or durations will usually be set implicitly by pliers, in some cases it makes sense to explicitly set a `Stim`'s temporal properties at initialization. For example, suppose we're processing log files from a psychology experiment where we presented a series of images to each subject in some predetermined temporal sequence. Even though each image is completely static, we may want pliers to keep track of the onset and duration of each image presentation with respect to the overall session. Assume we've processed our experiment log file to the point where we have a list of triples, with filename, onset, and duration of presentation as the elements in each tuple, respectively. Then we can easily construct a list of temporally aligned `ImageStim`s (for exegetical clarity, we use an explicit for-loop rather than a list comprehension):

```python

# A list of images, with each tuple representing
# the filename, onset, and duration, respectively.
image_list = [
    ('image1.jpg', 0, 2),
    ('image2.jpg', 3, 2),
    ('image3.jpg', 6, 2),
    ...
]

images = []
for (filename, onset, duration) in image_list:
    img = ImageStim(filename, onset=onset, duration=duration)
    images.append(img) 
```

If we now run the `images` list through one or more `ImageExtractor`s, the resulting `ExtractorResults` or pandas `DataFrame` (see the [ExtractorResult](#extractorresults) section below) will automatically log the correct onset and duration.

### Supported Stim types
At present, Pliers supports 4 primary types of stimuli: video files (`VideoStim`), audio files (`AudioStim`), images (`ImageStim`), and text (`TextStim`). Some of these `Stim` classes also have derivative or related subclasses that provide additional functionality. For example, the `ComplexTextStim` class internally stores a collection of `TextStim`s; most commonly, it is used to represent a sequence of words (e.g., as in a paragraph of coherent text), where each word is represented as a single `TextStim`, and the `ComplexTextStim` essentialy serves as a container that provides some extra tools (in particular, it provides useful methods for initializing multiple `TextStim`s from a single data source and retaining onset and duration information for each word). Similarly, a `VideoFrameStim` is a subclass of `ImageStim` that stores information about its location within an associated `VideoStim`.

### Intelligent Stim loading
In most cases, `Stim` instances don't have to be initialized directly by the user, as pliers will usually be able to infer the correct file type if filenames are passed directly to a `Transformer` (i.e., `my_extractor.transform('my_file.mp4')` will usually work, without first needing to do something like `stim = VideoStim('my_file.mp4)`). That said, it's generally a good idea to explicitly initialize one's `Stim`s, because (a) the initializers often take useful additional arguments, and (b) file-type detection is not completely foolproof, and (c) it makes your code clearer and more explicit.

As a sort of half-way approach that allows you to explicitly create a set of `Stim` objects, but doesn't require you to `import` a bunch of `Stim` classes, or even know what type of object you need, there's also a convenient `load_stims` method that attempts to intelligently guess the correct file type using python-magic (this is what pliers uses internally if you pass filenames directly to a `Transformer`). The following two examples produce identical results:

```python
### Approach 1: initialize each Stim directly
from pliers.stimuli import VideoStim, ImageStim, AudioStim, TextStim
stims = [
    VideoStim('my_video.mp4'),
    ImageStim('my_image.jpg'),
    AudioStim('my_audio.wav'),
    ComplexTextStim('my_text.txt')
]

# Approach 2: a shorter but potentially fallible alternative
from pliers.stimuli import load_stims
stims = load_stims(['my_video.mp4', 'my_image.jpg', 'my_audio.wav', 'my_text.txt'])
```
As you can see, mixed file types (and hence, heterogeneous lists of `Stim`s) are supported—though, again, we encourage you to use homogeneous lists wherever possible, for clarity.

### Transformation history
Although most `Stim` instances are initialized directly from a media file, in some cases, `Stim`s will be dynamically generated by applying a `Converter` or `Filter` to one or more other `Stim`s. For example, pliers implements several different API-based speech recognition converters, all of which take an `AudioStim` as input, and return a `ComplexTextStim` as output. Because the newly-generated `Stim` instance is not tied to a file, it's important to have some way of tracking its provenance. To this end, every `Stim` contains a `.history` attribute. The `history` is represented as an object of class `TransformationLog`, which is just a glorified `namedtuple`. It contains fields that track the name, class, and/or filename of the source (or original) `Stim`, the newly generated `Stim`, and the `Transformer` responsible for the transformation.

For example, this code:

```python
from pliers.converters import VideoToAudioConverter
from pliers.stimuli import VideoStim

video = VideoStim('../pliers/tests/data/video/obama_speech.mp4')
audio = VideoToAudioConverter().transform(video)
audio.history.to_df()
```

Produces this output:

    source_name                                      obama_speech.mp4
    source_file           ../pliers/tests/data/video/obama_speech.mp4
    source_class                                            VideoStim
    result_name                                      obama_speech.wav
    result_file           ../pliers/tests/data/video/obama_speech.wav
    result_class                                            AudioStim
    transformer_class                           VideoToAudioConverter
    transformer_params                                             {}

Here we only have a single row, but every transformation we perform will log will add another row to the `Stim`'s history. In cases with more than one transformatino, the `history` object will also have a `parent` field, which stores a reference to the previous transformation in the chain. This means that `TransformationLog` objects basically form a linked list that allows us to trace the full history of any `Stim` all the way back to some source file. If we ask for the string representation of a `history` object, we get a compact representation of the full trajectory. So, for example, if we load a `VideoStim` that then gets converted in turn to an `AudioStim` and then a `ComplexTextStim` (e.g., via speech-to-text transcription), the transformation history will look something like `'VideoStim->VideoToAudioConverter/AudioStim->IBMSpeechAPIConverter/ComplexTextStim'`.

## Transformers
As the name suggests, a `Transformer` is a kind of object that transforms other objects. In pliers, every `Transformer` always takes a single `Stim` as its input, though it can return different outputs. The `Transformer` API in pliers is modeled loosely on the widely-used [scikit-learn](http://scikit-learn.org/stable/) API; as such, what defines a `Transformer`, from a user's perspective, is that one can always call pass a `Stim` instance to `Transformer`'s `.transform()` method and expect to get another object as a result.

In practice, most users should never have any reason to directly instantiate the base `Transformer` class. We will almost invariably work with one of three different `Transformer` sub-classes: `Extractors`, `Converters`, and `Filters`. These classes are distinguished by the type of input that their respective `.transform` method expects, and the type of output that it produces:

| Transformer class    | Input    | Output          |
| -------------------: | :------: | :-------------: |
| Extractor            | AStim    | ExtractorResult |
| Converter            | AStim    | BStim           |
| Filter               | AStim    | AStim           |

Here, AStim and BStim are different `Stim` subclasses. So a `Converter` and `a Filter` are distinguished by the fact that a `Converter` always returns a different `Stim` class, while a `Filter` always returns a `Stim` of the same type as its input. This simple hierarchy turns out to be extremely powerful, as it enables us to operate in a natural, graph-like way over `Stims`, by filtering and converting them as needed before applying one or more `Extractors` to obtain extracted feature values.

Let's examine each of these `Transformer` types more carefully.

### Extractors
Extractors are the most important kind of `Transformer` in pliers, and in many cases, users will never have to touch any other kind of `Transformer` directly. Every `Extractor` implements a `transform()` instance method that takes a `Stim` object as its first argument, and returns an object of class `ExtractorResult` (see below). For example:

```
# Google Cloud Vision API face detection
from pliers.extractors import GoogleVisionAPIFaceExtractor

ext = GoogleVisionAPIExtractor()
result = ext.transform('my_image.jpg')
```

#### List of Extractor classes
Pliers is in the early days of development, so the list of available `Extractor`s is not very extensive at the moment. The `Extractor` classes that do exist mainly serve as a proof of concept, illustrating the range of potential tools and services that can be easily integrated into the package. We'll provide a comprehensive listing here in the near future; in the meantime you can inspect the `__all__` member of `pliers.extractors.__init__`, and then snoop around the codebase.

#### The ExtractorResult class
Calling `transform()` on an instantiated `Extractor` returns an object of class `ExtractorResult`. This is a lightweight container that contains all of the extracted feature information returned by the `Extractor`, and also stores references to the `Stim` and `Extractor` objects used to generate the result. The raw extracted feature values are stored in the `.data` property, but typically, we'll want to work with the data in a more  convenient format. Fortunately, every `ExtractorResult` instance exposes a `.to_df()` methods that gives us a nice pandas `DataFrame`. You can refer back to our first [Quickstart example](#example-the-first) to see this in action.

#### Merging Extractor results
In most cases, we'll want to do more than just apply a single `Extractor` to a single `Stim`. We might want to apply an `Extractor` to a set of stims (e.g., to run the `GoogleVisionAPIFaceExtractor` on a whole bunch of images), or to apply several different `Extractor`s to a single `Stim` (e.g., to run both face recognition and object recognition services on each image). As we'll see later (in the section on [Graphs](#graphs)), pliers makes it easy to apply many `Extractor`s to many `Stim`s, and in such cases, it will automatically merge the extracted feature data into one big pandas `DataFrame`. But in cases where we're working with multiple results manually, we can still merge the results ourselves, using the appropriately named `merge_results` function.

Suppose we have a list of images, and we want to run both face recognition and object labeling on each image. Then we can do the following:

```python
from pliers.extractors import (GoogleVisionAPIFaceExtractor, GoogleVisionAPILabelExtractor)
my_images = ['file1.jpg', 'file2.jpg', ...]

face_ext = GoogleVisionAPIFaceExtractor()
face_feats = face_ext.transform(my_images)

lab_ext = GoogleVisionAPILabelExtractor()
lab_feats = lab_ext.transform(my_images)
```

Now each of `face_feats` and `lab_feats` is a list of `ExtractorResult` objects. We could explicitly convert each element in each list to a pandas DataFrame (by calling `.to_df()`), but that would still be pretty unwieldy, as we would still need to figure out how to merge those DataFrames in a sensible way. Fortunately, `merge_results` will do all the work for us:

```python
from pliers.extractors import merge_results
# merge_results expects a single list, so we concatenate our two lists
df = merge_results(face_feats + lab_feats)
```

In the resulting DataFrame, every `Stim` is represented in a different row, and every feature is represented in a separate column. As noted earlier, the resulting DataFrame contains much more information than what's returned when we call `to_df()` on a single `ExtractorResult` object. Extra columns injected into the merged result include the name, class, filename (if any) and transformation history of each `Stim`; the name of each feature returned by each `Extractor`; and the name of each `Extractor` (as a second level in the column MultiIndex). You can prevent some of this additional information from being added by setting the `extractor_names` and `stim_names` arguments in `merge_results()` to `False` (by default, both are `True`).

### Converters
Converters, as their name suggests, *convert* `Stim` classes from one type to another. For example, the `IBMSpeechAPIConverter`, which is a subclass of `AudioToTextConverter`, takes an `AudioStim` as input, queries IBM's Watson speech-to-text API, and returns a transcription of the audio as a `ComplexTextStim` object. Most `Converter` classes have sensible names that clearly indicate what they do, but to prevent any ambiguity (and support type-checking), every concrete `Converter` class must define `_input_type` and `_output_type` properties that indicate what `Stim` classes they take and return as input and output, respectively.

#### Implicit Stim conversion
Although `Converter`s play a critical role in pliers, they usually don't need to be invoked explicitly by users, as pliers can usually figure out what conversions must be performed and carry them out implicitly. For example, suppose we want to run the `STFTAudioExtractor`---which computes the short-time Fourier transform on an audio clip and returns its power spectrum---on the audio track of a movie clip. We don't need to explicitly convert the `VideoStim` to an `AudioStim`, because pliers is clever enough to determine that it can get the appropriate input for the `STFTAudioExtractor` by executing the `VideoToAudioConverter`. In practice, then, the following two snippets produce identical results:

```python
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
```

Because pliers contains a number of "multistep" `Converter` classes, which chain together multiple standard `Converter`s, implicit Stim conversion will typically work not only for a single conversion, but also for a whole series of them. For example, if you feed a video file to a `LengthExtractor` (which just counts the number of characters in each `TextStim`'s text), pliers will use the built-in `VideoToTextConverter` class to transform your `VideoStim` into a `TextStim`, and everything should work smoothly in most cases.

I say "most" cases, because **there are two important gotchas to be aware of when relying on implicit conversion**. First, sometimes there's an inherent ambiguity about what trajectory a given stimulus should take through converter space; in such cases, the default conversions pliers performs may not line up with your expectations. For example, a `VideoStim` can be converted to a `TextStim` either by (a) extracting the audio track from the video and then transcribing into text via a speech recognition service, or (b) extracting the video frames from the video and then attempting to detect any text labels within each image. Because pliers has no way of knowing which of these you're trying to accomplish, it will default to the first. The upshot is that if you think there's any chance of ambiguity in the conversion process,  it's probably a good idea to explicitly chain the `Converter` steps (you can do this very easily using the [`Graph`](#graphs) interface discussed below). The explicit approach also provides additional precision in that you may want to initialize a particular `Converter` with non-default arguments, and/or specify exactly which of several candidate `Converter` classes to use (e.g., pliers defaults to performing speech-to-text conversion via the IBM Watson API, but also provides alternative support for the Wit.AI, and Google Cloud Speech APIs services).

Alternatively, you can set the default `Converter`(s) to use for any implicit `Stim` conversion at a package-wide level, via the `config.default_converters` attribute. By default, this is something like:

```python
default_converters = {
    'AudioStim->TextStim': ('IBMSpeechAPIConverter', 'WitTranscriptionConverter'),
    'ImageStim->TextStim': ('GoogleVisionAPITextConverter', 'TesseractConverter')
}
```

Here, each entry in the `default_converters` dictionary lists the `Converter`(s) to use, in order of preference. For example, the above indicates that any conversion between `ImageStim` and `TextStim` should first try to use the `GoogleVisionAPITextConverter`, and then, if that fails (e.g., because the user has no Google Cloud Vision API key set up), fall back on the `TesseractConverter`. If all selections specified in the config fail, pliers will still try to use any matching `Converter`s it finds, but you'll lose the ability to control the order of selection.

Second, because many `Converter`s call API-based services, if you're going to rely on implicit conversion, you should make sure that any API keys you might need are properly set up as environment variables in your local environment, seeing as you're not going to be able to pass those keys to the `Converter` as initialization arguments. For example, by default, pliers uses the IBM Watson API for speech-to-text conversion (i.e., when converting an `AudioStim` to a `ComplexTextStim`). But since you won't necessarily know this ahead of time, you won't be able to initialize the `Converter` with the correct credentials--i.e., by calling `IBMSpeechAPIConverter(username='my_username', password='my_password')`. Instead, the `Converter` will get initialized without any arguments (`IBMSpeechAPIConverter()`), which means the initialization logic will immediately proceed to look for IBM_USERNAME and IBM_PASSWORD variables in the environment, and will raise an exception if at least one of these variables is missing. So make sure as many API keys as possible are appropriately set in the environment. You can read more about this in the [API keys](#api-keys) section.

#### List of Converter classes
We'll have a more thorough listing and description of available `Converter` classes here in the near future. For now, you can scour the rest of this README, or have a look at the codebase.

### Filters
A `Filter` is a kind of `Transformer` that returns an object of the same `Stim` class as its input. For example, suppose you want to convert a color image to grayscale. In principle, one could easily write a `ColorToGrayScaleImageFilter` that fills this niche. In practice, there isn't much to be said about the `Filter` hierarchy at the moment, because we've hardly implemented any `Filter`s yet. So consider this a placeholder for the moment---and feel free to submit PRs with useful new `Filter` classes!

### Iterable-aware transformations
A useful feature of the `Transformer` API is that it's inherently iterable-aware: every pliers `Transformer` (including all `Extractors`, `Converters`, and `Filters`) can be passed an iterable (specifically, a list, tuple, or generator) of `Stim` objects rather than just a single `Stim`. The transformation will then be applied independently to each `Stim`.

### Caching and memory conservation
By default, pliers will cache  the output of all `.extract` calls to any `Transformer`. This can save an enormous amount of processing time, as it's very common to need to re-use converted `Stim` objects multiple times in a typical pliers workflow. However, this does have the potentially unwelcome side effect of ensuring all `Transformer` results are stored in memory (at the moment, pliers doesn't use disk-based caching). On a modern machine, this will rarely be a problem unless you're working with enormous files, but there may be cases where the default approach just isn't cutting it, and pliers' memory footprint gets too big. 
In such cases, you can take advantage of the fact that pliers internally uses generators rather than lists almost everywhere (and it's only the caching step that forces the conversion from the former to the latter). This means, for instance, that if you feed a video to an `Extractor` that expects an `ImageStim` input, the individual frames won't all be read into memory at conversion time; instead, a generator will be created, and each `VideoFrameStim` extracted from the `VideoStim` will only be instantiated when needed (a trick borrowed from the underlying [MoviePy](https://github.com/Zulko/moviepy/) library, which also iterates frames lazily). The upshot is that if you disable caching like so:

```python
from pliers import config
config.cache_transformers = False
```
...your memory footprint may decrease considerably—though probably at the cost of having to recompute many steps. A minor caveat to be aware of, when caching is off, is that certain `.transform()` calls may return generators rather than lists. Specifically, `Converter` classes will usually return generators when given lists as input, whereas `Extractor` classes should invariably return lists (because their outputs cannot be transformed any further). So on occasion, you may need to explicitly convert the result of a transformation to a list.

### API Keys
Many of the `Transformer`s in pliers rely on web-based APIs. For example, pliers includes support for face and object recognition via Google's Cloud Vision API, and enables conversion of audio files to text transcripts via several different speech-to-text services. While some of these APIs are free to use (and virtually all provide a limited number of free monthly calls), they all require each user to register for their own API credentials. This means that, in order to get the most out of pliers, you'll probably need to spend a few minutes registering accounts on a number of different websites. The following table lists all of the APIs supported by pliers at the moment, along with registration URLs:

| Transformer class | Web service | Environment variable(s) | Variable description |
| ----------------: | :---------: | :-------------------: | :---------------------: |
| WitTranscriptionConverter | [Wit.ai speech-to-text API](http://wit.ai)   | WIT_AI_API_KEY | Server Access Token |
| IBMSpeechAPIConverter | [IBM Watson speech-to-text API](https://www.ibm.com/watson/developercloud/speech-to-text.html) | IBM_USERNAME, IBM_PASSWORD | API username and password
| GoogleSpeechAPIConverter | [Google Cloud Speech API](https://cloud.google.com/speech/) | GOOGLE_APPLICATION_CREDENTIALS | path to .json discovery file |
| GoogleVisionAPITextConverter | [Google Cloud Vision API](https://cloud.google.com/vision/) | GOOGLE_APPLICATION_CREDENTIALS | path to .json discovery file |
| GoogleVisionAPIFaceExtractor | [Google Cloud Vision API](https://cloud.google.com/vision/) | GOOGLE_APPLICATION_CREDENTIALS | path to .json discovery file |
| GoogleVisionAPILabelExtractor | [Google Cloud Vision API](https://cloud.google.com/vision/) | GOOGLE_APPLICATION_CREDENTIALS | path to .json discovery file |
| GoogleVisionAPIPropertyExtractor | [Google Cloud Vision API](https://cloud.google.com/vision/) | GOOGLE_APPLICATION_CREDENTIALS | path to .json discovery file |
| IndicoAPIExtractor | [Indico.io API](https://indico.io) | INDICO_APP_KEY | API key |
| ClarifaiAPIExtractor | [Clarifai image recognition API](https://clarifai.com) | CLARIFAI_APP_ID, CLARIFAI_APP_SECRET | API app ID and secret |

Once you've obtained API keys for the services you intend to use, there are two ways to get pliers to recognize and use your credentials. First, each API-based `Transformer` can be passed the necessary values (or a path to a file containing those values) as arguments at initialization. For example:

```python
from pliers.extractors import ClarifaiAPIExtractor
ext = ClarifaiAPIExtractor(app_id='my_clarifai_app_id',
                           app_secret='my_clarifai_app_secret')
```

Alternatively, you can store the appropriate values as environment variables, in which case you can initialize a `Transformer` without any arguments. This latter approach is generally preferred, as it doesn't require you to hardcode potentially sensitive values into your code. The mandatory environment variable names for each service are listed in the table above.

```python
from pliers.extractors import GoogleVisionAPIFaceExtractor
# Works fine if GOOGLE_APPLICATION_CREDENTIALS is set in the environment
ext = GoogleVisionAPIFaceExtractor()
```

### Graphs
To this point, we've been initializing and running our `Transformer`s one at a time, and explicitly passing stimuli to each one. While this works fine, it can get rather verbose in cases where we want to extract a large number of features. It can also be a bit of a pain to appropriately connect `Converter`s to one another when the routing is complicated.

For example, suppose we have a series of videos (perhaps segments of a full-length movie) that contain on-screen subtitles, and we want to extract the subtitles from the image frames and run sentiment analysis on each chunk of extracted text. This requires us to (a) convert the `VideoStim` to a series of `VideoFrameStim`s, probably with some periodic sampling (there's no point in running text detection on every single frame, since subtitles won't change nearly that fast---we can probably get away with sampling frames as little as twice per second); (b) run text detection on each extracted frame (we'll use Google's Cloud Vision text detection API); and (c) apply one or more sentiment analysis `Extractor`s.

The code to do this, with all transformations made explicit:

```python
from pliers.stimuli import VideoStim
from pliers.converters import FrameSamplingConverter, GoogleVisionAPITextConverter
from pliers.extractors import (IndicoAPIExtractor, TensorFlowInceptionV3Extractor, merge_results)

# The input files
segments = ['segment1.mp4', 'segment2.mp4', 'segment3.mp4']
segments = [VideoStim(s) for s in segments]

### Initialize and chain converters ###
# Sample 2 video frames / second
frame_conv = FrameSamplingConverter(hertz=2)
frames = frame_conv.transform(segments)

# Run each image through Google's text detection API 
text_conv = GoogleVisionAPITextConverter()
texts = text_conv.transform(frames)

# Indico sentiment analysis extractor
indico = IndicoAPIExtractor()
sentiment_data = indico.transform(texts)

# Use TensorFlow Inception V3 model for image recognition
inception = TensorFlowInceptionV3Extractor()
label_data = inception.transform(frames)

# Merge into a single pandas DF
df = merge_results(results)
```
The above code listing is already pretty terse, and has the advantage of being explicit about every step. But if we want to save ourselves a few dozen keystrokes, we can use the `Graph` API to abbreviate the listing down to this:

```
from pliers.graph import Graph
from pliers.converters import FrameSamplingConverter

nodes = [
    (FrameSamplingConverter(hertz=2), [
        'TensorFlowInceptionV3Extractor',
        ('GoogleVisionAPITextConverter', ['IndicoAPIExtractor'])
    ])
]
g = Graph(nodes)
results = g.extract(segments)
```
At first glance, it may look like there's a lot of confusing nesting going on in the node definition, but it's actually not so bad. The key thing to recognize is that each node in the above graph is represented as a tuple with 2 elements. The first element is the `Transformer` to apply at that node, and the second contains any children nodes—i.e., nodes to which the output of the current node is passed. So, if we walk through the above example step by step, what we're saying is the following:

1. Define a root node that applies a `FrameSamplingConverter` to the input `Stim`(s), and passes the output to two children.
2. The first child node is simply specified as '`TensorFlowInceptionV3Extractor'`. Notice that because this node has no children of its own, we don't need to specify it as a tuple (but we could have equivalently written `('TensorFlowInceptionV3Extractor', [])`). This simply says that the node takes the input `Stim`, uses the Inception V3 model to label the input image(s), and returns the output.
3. The second child node applies the `GoogleVisionAPITextConverter` to the input received from the `FrameSamplingConverter`, and passes it along to its one child—a node containing an `IndicoAPIExtractor`.

Using this simple syntax, we can quickly construct `Graph`s with arbitrarily deep nestings. Note, once again, that we don't necessarily need to explicitly specify `Stim` conversion steps, as these will generally be detected and injected automatically (though, laziness aside, it's a good idea to be explicit, for reasons discussed earlier).

### Configuration
Pliers contains a number of package-wide settings that can be configured via the `pliers.config` module. These include:

Variable|Type|Default|Description
--------|----|-------|-----------
`cache_transformers`|`bool`|`True`|Whether or not to cache `Transformer` outputs in memory.
`log_transformations`|`bool`|`True`|Whether or not to log transformation details in each `Stim`'s `.history` attribute.
`drop_bad_extractor_results`|`bool`|`True`|When `True`, automatically removes any `None` values returned by any `Extractor`.
`progress_bar`|`bool`|`True`|Whether or not to display progress bars when looping over `Stim`s.
`default_converters`|`dict`|see module|See explanation inth [Converters](#implicit-stim-conversion) section.

These settings can be changed package-wide at run-time by setting new values in `config`; just make sure to import the `config` module itself rather than any of its members (or you'll import a static value, and changes won't propagate).

```python
...
from pliers import config
config.cache_transformers = False

# Caching is now off
extractor.transform(stim)
...
```
