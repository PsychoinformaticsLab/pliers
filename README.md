# *pliers*: a comprehensive framework for automated feature extraction

[![PyPI version fury.io](https://badge.fury.io/py/pliers.svg)](https://pypi.python.org/pypi/pliers/) ![pytest](https://github.com/PsychoinformaticsLab/pliers/actions/workflows/python-package.yml/badge.svg) [![Coverage Status](https://coveralls.io/repos/github/tyarkoni/pliers/badge.svg?branch=master)](https://coveralls.io/github/tyarkoni/pliers?branch=master)




## Overview

Pliers is a Python package for automated extraction of features from multimodal stimuli. It provides a unified, standardized interface to dozens of different feature extraction tools and services--including many state-of-the-art deep learning-based APIs. It's designed to let you rapidly and flexibly extract all kinds of useful information from videos, images, audio, and text.

You might benefit from pliers if you need to accomplish any of the following tasks (and many others!):

* Identify objects or faces in a series of images
* Transcribe the speech in an audio or video file
* Apply sentiment analysis to text
* Extract musical features from an audio clip
* Apply a part-of-speech tagger to a block of text

Each of the above tasks can typically be accomplished in 2 - 3 lines of code with pliers. Combining them *all*--and returning a single, standardized, integrated DataFrame as the result--might take a bit more work. Say maybe 5 or 6 lines.

In a nutshell, pliers provides an extremely high-level, unified interface to a very large number of feature extraction tools that span a wide range of modalities.

## How to cite
If you use pliers in your work, please cite both the pliers GitHub repository (http://github.com/tyarkoni/pliers) and the following paper:

> McNamara, Q., De La Vega, A., & Yarkoni, T. (2017, August). [Developing a comprehensive framework for multimodal feature extraction](https://dl.acm.org/citation.cfm?id=3098075). In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 1567-1574). ACM.

## Documentation

The official pliers documentation is quite thorough, and contains a comprehensive [quickstart](http://psychoinformaticslab.github.io/pliers/quickstart.html) doc (also available below), [user guide](http://psychoinformaticslab.github.io/pliers/) and complete [API Reference](http://psychoinformaticslab.github.io/pliers/reference.html).


## Installation

For the latest release:

> pip install pliers

Or, if you want to work on the bleeding edge:

> pip install pliers git+https://github.com/tyarkoni/pliers.git

### Dependencies
By default, installing pliers with pip will only install third-party libraries that are essential for pliers to function properly. These libraries are listed in requirements.txt. However, because pliers provides interfaces to a large number of feature extraction tools, there are literally dozens of other optional dependencies that may be required depending on what kinds of features you plan to extract (see optional-dependencies.txt). To be on the safe side, you can install all of the optional dependencies with pip:

> pip install -r optional-dependencies.txt

Note, however, that some of these Python dependencies have their own (possibly platform-dependent) requirements. For example, python-magic requires libmagic (see here for installation instructions), and without this, you’ll be relegated to loading all your stims explicitly rather than passing in filenames (i.e., `stim = VideoStim('my_video.mp4')` will work fine, but passing 'my_video.mp4' directly to an `Extractor` may not). Additionally, the Python OpenCV bindings require OpenCV3--but relatively few of the feature extractors in pliers currently depend on OpenCV, so you may not need to bother with this. Similarly, the `TesseractConverter` requires the tesseract OCR library, but no other `Transformer` does, so unless you’re planning to capture text from images, you’re probably safe.

### API Keys
While installing pliers itself is usually straightforward, setting up some of the web-based feature extraction APIs that pliers interfaces with can take a bit more effort. For example, pliers includes support for face and object recognition via Google’s Cloud Vision API, and enables conversion of audio files to text transcripts via several different speech-to-text services. While some of these APIs are free to use (and virtually all provide a limited number of free monthly calls), they all require each user to register for their own API credentials. This means that, in order to get the most out of pliers, you’ll probably need to spend some time registering accounts on a number of different websites. More details on API key setup are available [here](http://tyarkoni.github.io/pliers/installation.html#api-keys).
