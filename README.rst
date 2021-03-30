*pliers*: a python package for automated feature extraction
===========================================================

|PyPI version fury.io| |pytest| |Coverage Status|
|DOI:10.1145/3097983.3098075|

Pliers is a Python package for automated extraction of features from
multimodal stimuli. It provides a unified, standardized interface to
dozens of different feature extraction tools and services--including
many state-of-the-art deep learning-based models and content analysis
APIs. It's designed to let you rapidly and flexibly extract all kinds of
useful information from videos, images, audio, and text.

You might benefit from pliers if you need to accomplish any of the
following tasks (and many others!):

-  Identify objects or faces in a series of images
-  Transcribe the speech in an audio or video file
-  Apply sentiment analysis to text
-  Extract musical features from an audio clip
-  Apply a part-of-speech tagger to a block of text

Each of the above tasks can typically be accomplished in 2 - 3 lines of
code with pliers. Combining them *all*--and returning a single,
standardized DataFrame--might take a bit more
work. Say maybe 5 or 6 lines.

In a nutshell, pliers provides a high-level, unified interface to a
large number of feature extraction tools spanning a wide range of
modalities.

Documentation
-------------

The official pliers documentation is comprehensive, and contains a
`quickstart <http://psychoinformaticslab.github.io/pliers/quickstart.html>`__,
`user guide <http://psychoinformaticslab.github.io/pliers/>`__ and `API
Reference <http://psychoinformaticslab.github.io/pliers/reference.html>`__.

Installation
------------

Simply use pip to install the latest release:

   pip install pliers

Dependencies
~~~~~~~~~~~~

Installing pliers with pip will only install third-party
libraries that are essential for pliers to function properly. However,
because pliers provides interfaces to a large number of feature
extraction tools, there are dozens of optional dependencies that may be
required depending on what kinds of features you plan to extract. You
may install dependencies piece meal (pliers will alert you if
you're missing a depedency) or you may install all the required
dependencies:

   pip install -r optional-dependencies.txt

Note, that some of these Python dependencies may have their own requirements. 
For example, python-magic
requires libmagic and without this, you’ll be relegated to loading all
your stims explicitly rather than passing in filenames (i.e.,
``stim = VideoStim('my_video.mp4')`` will work fine, but passing
'my_video.mp4' directly to an ``Extractor`` may not).

Docker image
^^^^^^^^^^^^

You may also use the provided Docker image which fulfills all the optional dependencies.

:: 

   docker run -p 8888:8888 ghcr.io/psychoinformaticslab/pliers:unstable

Follow `these instructions <http://psychoinformaticslab.github.io/pliers/installation.html#docker>`__.

API Keys
^^^^^^^^

While installing pliers itself is straightforward, configuring web-based
feature extraction APIs can take a more
effort. For example, pliers includes support for face and object
recognition via Google’s Cloud Vision API, and enables conversion of
audio files to text transcripts via several different speech-to-text
services. While some of these APIs are free to use (and usually provide
a limited number of free monthly calls), they require users to
register to received API credentials. More details on API key setup
are available
`here <http://psychoinformaticslab.github.io/pliers/installation.html#api-keys>`__.

Another option is to exclusively use local models and algorithms, such as
the wide range covered by TensforFlow Hub using the ``TFHubExtractor``.

How to cite
-----------

If you use pliers in your work, please cite both the pliers and the following paper:

   McNamara, Q., De La Vega, A., & Yarkoni, T. (2017, August).
   `Developing a comprehensive framework for multimodal feature
   extraction <https://dl.acm.org/citation.cfm?id=3098075>`__. In
   Proceedings of the 23rd ACM SIGKDD International Conference on
   Knowledge Discovery and Data Mining (pp. 1567-1574). ACM.

.. |PyPI version fury.io| image:: https://badge.fury.io/py/pliers.svg
   :target: https://pypi.python.org/pypi/pliers/
.. |pytest| image:: https://github.com/PsychoinformaticsLab/pliers/actions/workflows/python-package.yml/badge.svg
.. |Coverage Status| image:: https://coveralls.io/repos/github/psychoinformaticslab/pliers/badge.svg?branch=master
   :target: https://coveralls.io/github/psychoinformaticslab/pliers?branch=master
.. |DOI:10.1145/3097983.3098075| image:: https://zenodo.org/badge/DOI/10.1145/3097983.3098075.svg
   :target: https://doi.org/10.1145/3097983.3098075
