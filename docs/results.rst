.. include:: _includes/_replacements.rst

Results
=======

Pliers is meant to provide a unified interface to a wide range of feature extraction tools and services, so it's imperative that the results we get back from different extractors can also be represented and accessed in a standardized way. In this section, we provide a detailed description of the pliers |ExtractorResult| class and the various options available for exporting results to pandas DataFrames. In many typical workflows, the export process will be done implicitly, so that all a user has to worry about is making sure to specify appropriate formatting arguments. In particular, if you only ever work with the :ref:`Graph API <graphs>`, you may want to skip down to the :ref:`Graph results` section.

The ExtractorResult class
-------------------------
Calling ``transform()`` on an instantiated |Extractor| returns an object of class |ExtractorResult|. This is a lightweight container that contains all of the extracted feature information returned by the |Extractor|, references to the |Stim| and |Extractor| objects used to generate the result, and both "raw" and processed forms of the results returned by the |Extractor| (though note that many Extractors don't set a ``.raw`` property). For example:

.. doctest::

    >>> from os.path import join
    >>> from pliers.tests.utils import get_test_data_path
    >>> jpg = join(get_test_data_path(), 'image', 'obama.jpg')
    >>> from pliers.extractors import FaceRecognitionFaceLocationsExtractor
    >>> ext = FaceRecognitionFaceLocationsExtractor()
    >>> result = ext.transform(jpg)
    >>> result.stim.name
    'obama.jpg'
    >>> result.extractor.name
    'FaceRecognitionFaceLocationsExtractor'
    >>> result.raw
    [(142, 349, 409, 82)]

.. _results-to-df:

Exporting results to pandas DataFrames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Typically, we'll want to work with the data in a more convenient form. Fortunately, every |ExtractorResult| instance provides a .to_df() method that returns a pandas DataFrame:

.. testsetup:: export_results

    from os.path import join
    from pliers.tests.utils import get_test_data_path
    jpg = join(get_test_data_path(), 'image', 'obama.jpg')
    from pliers.extractors import FaceRecognitionFaceLocationsExtractor
    ext = FaceRecognitionFaceLocationsExtractor()
    result = ext.transform(jpg)


.. doctest:: export_results
    :options: +NORMALIZE_WHITESPACE

    >>> result.to_df()
        order  duration  onset  object_id       face_locations
    0    NaN       NaN    NaN          0  (142, 349, 409, 82)

Here, the ``'face_locations'`` column is properly labeled with the name of the feature returned by the |Extractor|. Not surprisingly, you'll still need to know something about the feature extraction tool you're using in order to understand what you're getting back. In this case, consulting the documentation for the face_recognition package's `face_locations <http://pythonhosted.org/face_recognition/face_recognition.html#face_recognition.api.face_locations>`_ function reveals that the values ``(142, 349, 409, 82)`` give us the bounding box coordinates of the detected face in CSS order (i.e., top, right, bottom, left).

Timing columns
##############

You're probably wondering what the other columns are. The ``'onset'`` and ``'duration'`` columns providing timing information for the event in question, if applicable. In this case, because our source |Stim| was a static image, there's no meaningful timing information to be had. But ``to_df()`` still returns these columns by default. This becomes important in cases where we want to preserve some temporal context as we pass |Stim| objects through a feature extraction pipeline:

.. doctest::
    :options: +NORMALIZE_WHITESPACE
     
    >>> from os.path import join
    >>> from pliers.tests.utils import get_test_data_path
    >>> jpg = join(get_test_data_path(), 'image', 'obama.jpg')
    >>> from pliers.extractors import FaceRecognitionFaceLocationsExtractor
    >>> from pliers.stimuli.image import ImageStim
    >>> ext = FaceRecognitionFaceLocationsExtractor()
    >>> image = ImageStim(jpg, onset=14, duration=1)
    >>> result = ext.transform(image)
    >>> result.to_df()
    order  duration  onset  object_id       face_locations
    0    NaN         1     14          0  (142, 349, 409, 82)

Of course, if we really don't want the timing columns, we can easily suppress them:

.. testsetup:: timing

    from os.path import join
    from pliers.tests.utils import get_test_data_path
    jpg = join(get_test_data_path(), 'image', 'obama.jpg')
    from pliers.extractors import FaceRecognitionFaceLocationsExtractor
    from pliers.stimuli.image import ImageStim
    ext = FaceRecognitionFaceLocationsExtractor()
    image = ImageStim(jpg, onset=14, duration=1)
    result = ext.transform(image)

.. doctest:: timing
    :options: +NORMALIZE_WHITESPACE

    >>> result.to_df(timing=False)
       object_id       face_locations
    0          0  (142, 349, 409, 82)

We could also pass ``timing='auto'``, which would drop the ``'onset'`` and ``'duration'`` columns if and only if all values are ``NaN``.

Understanding object_ids
########################

What about the ``'object_id'`` column? This one's not so intuitive, but can in some cases be very important. Consider a situation where there are multiple valid results objects in a single |Stim|. For example, suppose we feed an image to our |FaceRecognitionFaceLocationsExtractor| that contains multiple faces. How are we supposed to keep track of these different faces in the results? They come from the same |Stim| and share the same timing parameters (e.g., in the last example, where we explicitly specified the onset and duration, all detected faces will have ``onset=14`` and ``duration=1``). But we obviously need to have *some* way of distinguishing distinct records.

The solution is to serially assign each distinct result object a different ``object_id``. Let's modify the last example to feed in an image that contains 4 separate faces:

.. doctest::
    :options: +NORMALIZE_WHITESPACE

    >>> from os.path import join
    >>> from pliers.tests.utils import get_test_data_path
    >>> jpg = join(get_test_data_path(), 'image', 'thai_people.jpg')
    >>> from pliers.extractors import FaceRecognitionFaceLocationsExtractor
    >>> from pliers.stimuli.image import ImageStim
    >>> ext = FaceRecognitionFaceLocationsExtractor()
    >>> image = ImageStim(jpg, onset=14, duration=1)
    >>> result = ext.transform(image)
    >>> result.to_df()
       order  duration  onset  object_id        face_locations
    0    NaN         1     14          0  (236, 862, 325, 772)
    1    NaN         1     14          1  (104, 581, 211, 474)
    2    NaN         1     14          2  (365, 782, 454, 693)
    3    NaN         1     14          3  (265, 444, 355, 354)

As with the ``timing`` columns, if we don't want to see the ``object_id`` column, we can suppress it by calling ``.to_df(object_id=False)`` or ``.to_df(object_id='auto')``. In the latter case, the ``object_id`` column will be included if and only if the values are non-constant (i.e., there is some value other than 0 somewhere in the DataFrame).

Displaying metadata
###################

Although not displayed by default, it's also possible to include additional metadata about the |Stim| and |Extractor| in the DataFrame returned by ``to_df``:


.. doctest:: timing
    :options: +NORMALIZE_WHITESPACE

    >>> result = ext.transform(jpg)
    >>> result.to_df(timing=False, object_id=False, metadata=True)
        face_locations  stim_name      class                                 filename history                              source_file
    0  (142, 349, 409, 82)  obama.jpg  ImageStim  ...obama.jpg          ...obama.jpg

Here we get columns for the |Stim| name (typically just the filename, unless we explicitly specified a different name), the current filename, the |Stim| history, and the source filename. In the above example, ``stim_name``, ``filename`` and ``source_file`` are identical, but this won't always be the case. For example, if the images we're running through the |FaceRecognitionFaceLocationsExtractor| had been extracted from frames of video, the ``source_file`` would point to the original video, while the ``filename`` would point to (temporary) image files corresponding to the extracted frames.

The ``history`` column contains a text summary of the |Stim| transformation history; for more details, see the :ref:`Transformation history` section.

Display mode
############
By default, DataFrames are in 'wide' format. That is, each row represents a single event, and all features are contained in columns. To get a better sense of what this means, it's helpful to look at an extractor that returns more than one feature:

.. doctest::
    :skipif: os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None
    :options: +NORMALIZE_WHITESPACE

    >>> from os.path import join
    >>> from pliers.tests.utils import get_test_data_path
    >>> apple = join(get_test_data_path(), 'image', 'apple.jpg')
    >>> from pliers.extractors import GoogleVisionAPILabelExtractor
    >>> ext = GoogleVisionAPILabelExtractor()
    >>> result = ext.transform(apple)
    >>> result.to_df()
    onset   duration    object_id   fruit   apple   produce food    natural foods   mcintosh    diet food
    NaN     NaN         0           0.968   0.966   0.959   0.824   0.801           0.629   0.607

Here we fed in an image of an apple, and the |GoogleVisionAPILabelExtractor| automatically returned 7 different high-probability labels ('fruit', 'apple', etc.)--each one represented as a separate feature in our results.

While there's nothing at all wrong with this format (indeed, it's the default!), sometimes we prefer to get back our data in 'long' format, where each row represents the intersection of a single event and a single feature:

.. testsetup:: display
    :skipif: os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None

    from os.path import join
    from pliers.tests.utils import get_test_data_path
    apple = join(get_test_data_path(), 'image', 'apple.jpg')
    from pliers.extractors import GoogleVisionAPILabelExtractor
    ext = GoogleVisionAPILabelExtractor()
    result = ext.transform(apple)

.. doctest:: display
    :skipif: os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None

    >>> result.to_df(format='long', timing=False, object_id=False)
    feature         value
    fruit           0.968
    apple           0.966
    produce         0.959
    food            0.824
    natural foods   0.801
    mcintosh        0.629
    diet food       0.607

Now, the feature names are specified in the ``'feature'`` column, and the extracted values are provided in the ``'value'`` column.

Displaying Extractor names
##########################

If we only ever worked with results generated by a single |Extractor| for a single |Stim|, we'd rarely have any problems figuring out where our results are coming from. But as we'll see momentarily, a more common use case is that we want to combine results from multiple Extractors and/or Stims into a single, possibly very large, DataFrame. In this case, figuring out the source of particular features can quickly get confusing--especially because different Extractors can potentially have similar or even identical feature names.

We can ensure that the name of the current |Extractor| is explicitly added to our results via the ``extractor_name`` argument. The precise behavior of ``extractor_name=True`` will depend on the ``format`` argument. When ``format='wide'``, the name will be added as the first level in a pandas MultiIndex; when ``format='long'``, a new column will be added. Examples:

.. doctest:: display
    :skipif: os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None

    >>> results.to_df(format='long', timing=False, object_id=False, extractor_name=True)
    feature         value   extractor
    fruit           0.968   GoogleVisionAPILabelExtractor
    apple           0.966   GoogleVisionAPILabelExtractor
    produce         0.959   GoogleVisionAPILabelExtractor
    food            0.824   GoogleVisionAPILabelExtractor
    natural foods   0.801   GoogleVisionAPILabelExtractor
    mcintosh        0.629   GoogleVisionAPILabelExtractor
    diet food       0.607   GoogleVisionAPILabelExtractor
    >>> results.to_df(timing=False, object_id=False, extractor_name=True)
    GoogleVisionAPILabelExtractor
    fruit   apple   produce food    natural foods   mcintosh    diet food
    0.968   0.966   0.959   0.824   0.801           0.629       0.607

Merging Extractor results
-------------------------
In most cases, we'll want to do more than just apply a single |Extractor| to a single |Stim|. We might want to apply one |Extractor| to a set of stims, several different Extractors to a single |Stim|, or many Extractors to many Stims. As described (in the section on :ref:`graphs`, pliers makes it easy to build such workflows--and to automatically merge the extracted feature data into one big pandas DataFrame. But in cases where we're working with multiple results manually, or wish to exercise a little more control over the output format, we can still merge the results ourselves, using the appropriately named ``merge_results`` function.

Suppose we have a list of images, and we want to run both face recognition and object labeling on each image. Then we can do the following:

::

    from pliers.extractors import (GoogleVisionAPIFaceExtractor, GoogleVisionAPILabelExtractor)
    my_images = ['file1.jpg', 'file2.jpg', ...]

    face_ext = GoogleVisionAPIFaceExtractor()
    face_feats = face_ext.transform(my_images)

    lab_ext = GoogleVisionAPILabelExtractor()
    lab_feats = lab_ext.transform(my_images)

Now each of ``face_feats`` and ``lab_feats`` is a list of |ExtractorResult| objects. We could explicitly convert each element in each list to a pandas DataFrame (by calling ``.to_df()``), but then we'd still need to figure out how to merge all those DataFrames in a sensible way. Fortunately, :func:`merge_results` can do the heavy lifting for us:

::

    from pliers.extractors import merge_results
    # merge_results expects a single list, so we concatenate our two lists
    df = merge_results(face_feats + lab_feats, timing=False, metadata=True,
                       object_id='auto', format='long',
                       extractor_names='column')

Nearly all of the arguments to ``merge_results`` match the ones :ref:`we saw above<results-to-df>` when calling ``to_df`` on individual |ExtractorResult| instances. We can control the output shape ('wide' vs. 'long') with the ``format`` argument, and indicate whether which optional columns to include via the ``metadata``, ``timing``, and ``object_id`` flags.

The only notable exception in terms of argument behavior is that the ``extractor_names`` argument has different semantics from ``extractor_name`` in ``to_df()``. Specifically, rather than just specifying whether or not to include the names of Extractors, we can now also control exactly how they're displayed (e.g., whether they're prepended to the feature names, added as a level in a pandas MultiIndex, etc.). Full details can be found in the :func:`merge_results` function reference.

In all other respects, the outputs of ``merge_results`` should look just like those generated by ``to_df`` calls--except of course that the results for different Extractors and Stims are now concatenated together along either the row or the column axes (depending on the ``format`` argument). As a general rule of thumb, we recommend using the default format ('wide') in cases where one is working with a small number of different Extractors and/or features, and switching to ``format='long'`` when the number of Extractors and/or features gets large. (The main reason for this recommendation is that the merged DataFrames are typically sparse, so in 'wide' format, one can end up with a very large number of ``NaN`` values when working with many Extractors at once. In 'long' format, there are virtually no missing values.)

Graph results
-------------
In practice, many users will primarily rely on the :ref:`Graph API <graphs>` for feature extraction. Since standard |Graph| execution merges results by default, using a |Graph| means that you probably won't need to worry about calling ``to_df`` or ``merge_results`` explicitly. You'll just get a single, already-merged pandas DataFrame as the result of a ``Graph.transform`` call.

The main thing to be aware of in this case is that the ``.transform`` call takes any of the keyword arguments supported by ``merge_results``, and simply passes them through. This means you can control the output format and inclusion of various columns exactly as documented above for ``merge_results`` (and ``to_df``). Here's a minimalistic example to illustrate:


.. doctest::

    from pliers.graph import Graph
    from pliers.filters import FrameSamplingFilter
    from pliers.extractors import FaceRecognitionFaceLandmarksExtractor
    from pliers.tests.utils import get_test_data_path
    from os.path import join

    # Use a short video from the pliers test suite
    video = join(get_test_data_path(), 'video', 'obama_speech.mp4')

    # Sample image frames from video, then apply face recognition
    nodes = [
        (FrameSamplingFilter(hertz=2), ['FaceRecognitionFaceLocationsExtractor'])
    ]

    # Construct and run Graph
    g = Graph(nodes)
    df = g.transform(video, metadata=False, format='long', extractor_names='column')

    ''' Outputs:

    object_id   duration    onset   feature         value               extractor
    0           0.50        0.0     face_locations  (36, 223, 79, 180)  FaceRecognitionFaceLocationsExtractor
    1           0.50        0.0     face_locations  (58, 101, 94, 65)   FaceRecognitionFaceLocationsExtractor
    0           0.50        0.5     face_locations  (26, 213, 70, 170)  FaceRecognitionFaceLocationsExtractor
    1           0.50        0.5     face_locations  (58, 101, 94, 65)   FaceRecognitionFaceLocationsExtractor
    ...         ...         ...     ...             ...                 ...


If you don't explicitly specify any formatting arguments, you'll get the same sane defaults used in :py:func:`merge_results`, which should work well in most cases.
