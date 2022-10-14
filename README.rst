*pliers*: a python package for automated feature extraction
===========================================================

|PyPI version fury.io| |pytest| |Coverage Status| |docs|
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

Pliers overview (with application to naturalistic fMRI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pliers is a general purpose tool, this is just one domain where it's useful.

.. raw:: html

   <iframe width="560" height="315" src="https://www.youtube.com/embed/4mQjtyQPu_c" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

`Tutorial Video <https://www.youtube.com/watch?v=4mQjtyQPu_c>`__

The above video is from a `tutorial <https://naturalistic-data.org/content/Pliers_Tutorial.html>`__
as a part of a `course about naturalistic data <https://naturalistic-data.org/>`__.

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
.. |docs| image:: https://readthedocs.org/projects/pliers/badge/?version=latest
    :target: https://pliers.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
