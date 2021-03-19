*pliers*: a python package for automated feature extraction
===========================================================

|PyPI version fury.io| |pytest| |Coverage Status|
|DOI:10.1145/3097983.3098075|

Pliers is a Python package for automated extraction of features from
multimodal stimuli. It provides a unified, standardized interface to
dozens of different feature extraction tools and services–including many
state-of-the-art deep learning-based models and content analysis APIs.
It’s designed to let you rapidly and flexibly extract all kinds of
useful information from videos, images, audio, and text.

You might benefit from pliers if you need to accomplish any of the
following tasks (and many others!):

-  Identify objects or faces in a series of images
-  Transcribe the speech in an audio or video file
-  Apply sentiment analysis to text
-  Extract musical features from an audio clip
-  Apply a part-of-speech tagger to a block of text

Each of the above tasks can typically be accomplished in 2 - 3 lines of
code with pliers. Combining them *all*–and returning a single,
standardized, integrated DataFrame as the result–might take a bit more
work. Say maybe 5 or 6 lines.

In a nutshell, pliers provides a high-level, unified interface to a very
large number of feature extraction tools that span a wide range of
modalities.

Documentation
-------------

The official pliers documentation is quite thorough, and contains a
comprehensive `quickstart`_, `user guide`_ and complete `API
Reference`_.

Installation
------------

Simply use pip to install the latest release:

   pip install pliers

Dependencies
~~~~~~~~~~~~

By default, installing pliers with pip will only install third-party
libraries that are essential for pliers to function properly. However,
because pliers provides interfaces to a large number of feature
extraction tools, there are dozens of optional dependencies that may be
required depending on what kinds of features you plan to extract. You
can choose to install dependencies piece meal (pliers will alert you if
you’re missing a depedency) or you may install all the required
dependencies:

   pip install -r optional-dependencies.txt

Note, however, that some of these Python dependencies have their own
(possibly platform-dependent) requirements. For example, python-magic
requires libmagic and without this, you’ll be relegated to loading all
your stims explicitly rather than passing in filenames (i.e.,
``stim = VideoStim('my_video.mp4')`` will work fine, but passi

.. _quickstart: http://psychoinformaticslab.github.io/pliers/quickstart.html
.. _user guide: http://psychoinformaticslab.github.io/pliers/
.. _API Reference: http://psychoinformaticslab.github.io/pliers/reference.html

.. |PyPI version fury.io| image:: https://badge.fury.io/py/pliers.svg
   :target: https://pypi.python.org/pypi/pliers/
.. |pytest| image:: https://github.com/PsychoinformaticsLab/pliers/actions/workflows/python-package.yml/badge.svg
.. |Coverage Status| image:: https://coveralls.io/repos/github/tyarkoni/pliers/badge.svg?branch=master
   :target: https://coveralls.io/github/tyarkoni/pliers?branch=master
.. |DOI:10.1145/3097983.3098075| image:: https://zenodo.org/badge/DOI/10.1145/3097983.3098075.svg
   :target: https://doi.org/10.1145/3097983.3098075