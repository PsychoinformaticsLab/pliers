Stims
=====

.. include:: _includes/_replacements.rst


The core data object used in pliers is something called a |Stim| (short for stimulus). A |Stim| instance is a lightweight wrapper around any one of several standard file or data types: a video file, an image, an audio file, or some text. Under the hood, pliers uses other Python libraries to support most operations on the original files (e.g., MoviePy for video and audio files, and pillow/PIL for images).

Initialization
--------------
We can minimally initialize new |Stim| instances by passing in a filename to the appropriate class initializer. Alternatively, some |Stim| types also accept a data argument that allows initialization from an appropriate data type instead of a file (e.g., a numpy array for images, a string for text, etc.). Examples:

::

    from pliers.stimuli import VideoStim, ImageStim, TextStim

    # Initialize from files
    vs = VideoStim('my_video_file.mp4')
    img1 = ImageStim('first_image.jpg')

    # Initialize a TextStim from a string
    text = TextStim("this sentence will be represented as a single token of text")

    # Initialize an ImageStim from a random RGB array
    img_data = np.random.uniform(size=(100, 100, 3))
    img2 = ImageStim(data=img_data)

In general, users will rarely have a need to directly manipulate |Stim| classes, beyond them passing them around to |Extractor|\s. In fact, it's often not necessary to initialize |Stim|\s at all, as you can generally pass a list of string filenames to any |Extractor| (with some caveats discussed below).

All Stim classes expose a number of attributes that provide some information about the associated stimulus. Most notably, the :py:`.filename` and :py:`.name` attributes store the names of the source file (e.g., my_movie.mp4) and an (optional) derived name, respectively.

Types of Stims
--------------
At present, Pliers supports 4 primary types of stimuli: video files (|VideoStim|), audio files (AudioStim), images (|ImageStim|), and text (|TextStim|). Some of these Stim classes also have derivative or related subclasses that provide additional functionality. For example, the |ComplexTextStim| class internally stores a collection of |TextStim|\s; most commonly, it is used to represent a sequence of words (e.g., as in a paragraph of coherent text), where each word is represented as a single TextStim, and the |ComplexTextStim| essentialy serves as a container that provides some extra tools (in particular, it provides useful methods for initializing multiple TextStims from a single data source and retaining onset and duration information for each word). Similarly, a |VideoFrameStim| is a subclass of |ImageStim| that stores information about its location within an associated |VideoStim|.

The current |Stim| hierarchy in pliers (ignoring |CompoundStim| classes discussed below) includes:

.. currentmodule:: pliers.stimuli
.. autosummary::

    AudioStim
    ImageStim
    TextStim
    VideoStim
    VideoFrameStim
    VideoFrameCollectionStim
    ComplexTextStim

CompoundStim classes
~~~~~~~~~~~~~~~~~~~~
In many real-world applications, we need to extract features from objects more complex than just text, audio, or images. To facilitate analysis of a wider range of objects, pliers defines a |CompoundStim| class that serves as a unified container for any number of other |Stim| classes. Every |CompoundStim| class defines which |Stim| classes it's allowed to contain as components. These components are automatically available as attributes of the |CompoundStim| with standard Python variable names based on the class names (e.g., |ImageStim| and |ComplexTextStim| components would be available at :py:`.image` and :py:`.complex_text`, respectively).

For example, pliers defines a |TranscribedAudioCompoundStim| that's used to represent an audio clip that's been transcribed. This object internally stores both an |AudioStim| (containing the audio) and a |ComplexTextStim| (containing the transcribed text tokens and associated timing information). A |TranscribedAudioCompoundStim| is initialized by passing in exactly one of each of these components (e.g., :py:`tac = TranscribedAudioComponentStim(audio, text)`).

One of the nice features of |CompoundStim| classes is that any pliers |Transformer| will automatically detect and process matching components inside the |CompoundStim|. For example, the |TranscribedAudioCompoundStim| can be passed to any |Extractor| that accepts either an |AudioStim| or a |ComplexTextStim| as input. This means that we don't have to worry about specifying complicated rules for determining what input to provide to Transformers; in most cases, we can get the intended behavior just by naively passing |Stim|\s to a |Transformer|.

Consider this code:

::

    from pliers.stimuli import TranscribedAudioCompoundStim
    from pliers.graph import Graph

    # assume we have audio and text Stims we're passing in
    tac = TranscribedAudioCompoundStim(audio, text)

    # Construct a Graph with two extractors
    graph = Graph(['PartOfSpeechExtractor', 'RMSExtractor'])

    # Apply the extractors to the Stim
    result = graph.transform(tac)

The above code should complete gracefully, returning a pandas DataFrame that combines the part-of-speech information extracted from the text component of the |TranscribedAudioCompoundStim| with root-mean-square energy extracted from the audio component.

Existing classes derived from |CompoundStim| include:

.. currentmodule:: pliers.stimuli
.. autosummary::

    CompoundStim
    TranscribedAudioCompoundStim
    TweetStim

Beyond these predefined classes, new |CompoundStim| classes are usually very easy to define. For example, here's the entire definition of the |TranscribedAudioCompoundStim| in pliers:

::

    class TranscribedAudioCompoundStim(CompoundStim):

        _allowed_types = (AudioStim, ComplexTextStim)
        _allow_multiple = False
        _primary = AudioStim

        def __init__(self, audio, text):
            super(TranscribedAudioCompoundStim, self).__init__(elements=[audio, text])

As you can see, not much work is required. All we have to do is indicate (a) which |Stim| types are allowed as components of the |CompoundStim|, (b) whether or not multiple instances of the same type are allowed (in this case, it doesn't make sense to allow a transcribed audio clip to contain multiple audio clips or text transcriptions), and (c) what the primary component type is (which is used to define the name and original filename of the |CompoundStim| as a whole).

Intelligent loading
-------------------
In most cases, Stim instances don't have to be initialized directly by the user, as pliers will usually be able to infer the correct file type if filenames are passed directly to a |Transformer| (i.e., my_extractor.transform('my_file.mp4') will usually work, without first needing to do something like stim = VideoStim('my_file.mp4)). That said, it's generally a good idea to explicitly initialize one's Stims, because the initializers often take useful additional arguments, and file-type detection is also not completely foolproof.

As a sort of half-way approach that allows you to explicitly create a set of Stim objects, but doesn't require you to import a bunch of Stim classes, or even know what type of object you need, there's also a convenient load_stims method that attempts to intelligently guess the correct file type using python-magic (this is what pliers uses internally if you pass filenames directly to a |Transformer|). The following two examples produce identical results:

::

    # Approach 1: initialize each Stim directly
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

As you can see, mixed file types (and hence, heterogeneous lists of Stims) are supported--though, again, we encourage you to use homogeneous lists wherever possible, for clarity.

Iterable Stims
--------------

Some |Stim| classes are naturally iterable, so pliers makes it easy to iterate their elements. For example, a |VideoStim| is made up of a series of |VideoFrameStim|\s, and a |ComplexTextStim| is made up of |TextStim|\s. Looping over the constituent elements is trivial:

::

    >>> from pliers.stimuli import ComplexTextStim
    >>> cts = ComplexTextStim(text="This class method uses the default nltk word tokenizer, which will split this sentence into a list of TextStims, each representing a single word.")
    >>> for ts in cts:
    >>>    print(ts.text)
    "This"
    "class"
    "method"
    ...

You can also directly access the constituent elements within the containing Stim if you need to (e.g., :py:`VideoStim.frames` or :py:`ComplexTextStim.elements`). Note that, for efficiency reasons, these properties will typically return generators rather than lists (e.g., retrieving the :py:`.frames` property of a |VideoStim| will return a generator that reads frames lazily). You can always explicitly convert the generator to a list (e.g., :py:`frames = list(video.frames)`), just be aware that your memory footprint may instantly balloon in cases where you're working with large media files.

Timing information
------------------
Some |Stim| classes inherently have a temporal dimension (e.g., |VideoStim| and |AudioStim|). However, even a static |Stim| instance such as an |ImageStim| or a |TextStim| will often be assigned :py:`.onset` and :py:`.duration` properties during initialization. Typically, this happens because the static |Stim| is understood to be embedded within some temporal context. Consider the following code:

::

    >>> vs = VideoStim('my_video.mp4')
    >>> frame_200 = vs.get_frame(200)
    >>> print(frame_200.onset, frame_200.duration)
    6.666666666666667, 0.03333333333333333

Assume the above video runs at ~30 frames per second. Then, when we retrieve the 200th video frame, the result is a |VideoFrameStim| that, in addition to storing a numpy array in its :py:`.data` attribute (i.e., the actual image content of the frame), also keeps track of (a) the onset of the frame (in seconds) relative to the start of the source video file, and (b) the duration for which the |VideoFrameStim| is supposed to be presented (in this case 1/30 seconds).

Although |Stim| onsets and/or durations will usually be set implicitly by pliers, in some cases it makes sense to explicitly set a |Stim|'s temporal properties at initialization. For example, suppose we're processing log files from a psychology experiment where we presented a series of images to each subject in some predetermined temporal sequence. Even though each image is completely static, we may want pliers to keep track of the onset and duration of each image presentation with respect to the overall session. Assume we've processed our experiment log file to the point where we have a list of triples, with filename, onset, and duration of presentation as the elements in each tuple, respectively. Then we can easily construct a list of temporally aligned ImageStims:

::

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

If we now run the images list through one or more ImageExtractors, the resulting ExtractorResults or pandas DataFrame will automatically log the correct onset and duration.

Transformation history
----------------------
Although most |Stim| instances are initialized directly from a media file, in some cases, |Stim|\s will be dynamically generated by applying a Converter or Filter to one or more other |Stim|\s. For example, pliers implements several different API-based speech recognition converters, all of which take an |AudioStim| as input, and return a |ComplexTextStim| as output. Because the newly-generated |Stim| instance is not tied to a file, it's important to have some way of tracking its provenance. To this end, every |Stim| contains a .history attribute. The history is represented as an object of class TransformationLog, which is just a glorified namedtuple. It contains fields that track the name, class, and/or filename of the source (or original) |Stim|, the newly generated |Stim|, and the |Transformer| responsible for the transformation.

For example, this code:

::

    from pliers.converters import VideoToAudioConverter
    from pliers.stimuli import VideoStim

    video = VideoStim('../pliers/tests/data/video/obama_speech.mp4')
    audio = VideoToAudioConverter().transform(video)
    audio.history.to_df()

Produces this output:

================== =====
source_name        obama_speech.mp4
source_file        ../pliers/tests/data/video/obama_speech.mp4
source_class       VideoStim
result_name        obama_speech.wav
result_file        ../pliers/tests/data/video/obama_speech.wav
result_class       AudioStim
transformer_class  VideoToAudioConverter
transformer_params {}
================== =====

Here we only have a single row (in the above table, the first row contains what would normally be displayed as the column names), but every transformation we perform will log another row in the |Stim|'s history. In cases with more than one transformation, the history object will also have a parent field, which stores a reference to the previous transformation in the chain. This means that TransformationLog objects basically form a linked list that allows us to trace the full history of any |Stim| all the way back to some source file. If we ask for the string representation of a history object (i.e., :py:`str(history)`), we get a compact representation of the full trajectory. So, for example, if we load a |VideoStim| that then gets converted in turn to an |AudioStim| and then a |ComplexTextStim| (e.g., via speech-to-text transcription), the transformation history will look something like 'VideoStim->VideoToAudioConverter/AudioStim->IBMSpeechAPIConverter/ComplexTextStim'.
