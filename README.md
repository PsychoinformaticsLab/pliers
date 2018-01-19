# pliers

A Python package for automated feature extraction.

## Status
* [![Build Status](https://travis-ci.org/tyarkoni/pliers.svg?branch=master)](https://travis-ci.org/tyarkoni/pliers)
* [![Coverage Status](https://coveralls.io/repos/github/tyarkoni/pliers/badge.svg?branch=master)](https://coveralls.io/github/tyarkoni/pliers?branch=master)

## Overview

Pliers is a Python package for automated extraction of features from multimodal stimuli. It's designed to let you rapidly and flexibly extract all kinds of useful information from videos, images, audio, and text. It's also easily extensible, allowing users to write new feature extractors in relatively little code, and providing a common framework for interfacing with all kinds of domain-specific tools.

## Installation

For the latest release:

> pip install pliers

Or, if you want to work on the bleeding edge:

> pip install pliers git+https://github.com/tyarkoni/pliers.git

### Dependencies

By default, installing pliers with pip will only install third-party libraries that are essential for pliers to function properly. These libraries are listed in [requirements.txt](https://github.com/tyarkoni/pliers/blob/master/requirements.txt). However, because pliers provides interfaces to a large number of feature extraction tools, there are literally dozens of other optional dependencies that may be required depending on what kinds of features you plan to extract (see [optional-dependencies.txt](https://github.com/tyarkoni/pliers/blob/master/optional-dependencies.txt)). To be on the safe side, you can install all of the optional dependencies with pip:

> pip install -r optional-dependencies.txt

Note, however, that some of these Python dependencies have their own (possibly platform-dependent) requirements. Most notably, [python-magic](https://github.com/ahupp/python-magic) requires libmagic (see [here](https://github.com/ahupp/python-magic#dependencies) for installation instructions), and without this, you’ll be relegated to loading all your stims explicitly rather than passing in filenames (i.e., `stim = VideoStim(‘my_video.mp4’)` will work fine, but passing `‘my_video.mp4’` directly to an `Extractor` will not). Additionally, the Python OpenCV bindings require [OpenCV3](http://opencv.org/) (which can be a bit more challenging to [install](http://docs.opencv.org/2.4/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html))–but relatively few of the feature extractors in pliers currently depend on OpenCV, so you may not need to bother with this. Similarly, the TesseractConverter requires the tesseract OCR library, but no other Transformer does, so unless you’re planning to capture text from images, you’re probably safe.

### API Keys
While installing pliers itself is usually straightforward, setting up some of the web-based feature extraction APIs that pliers interfaces with can take a bit more effort. For example, pliers includes support for face and object recognition via Google’s Cloud Vision API, and enables conversion of audio files to text transcripts via several different speech-to-text services. While some of these APIs are free to use (and virtually all provide a limited number of free monthly calls), they all require each user to register for their own API credentials. This means that, in order to get the most out of pliers, you’ll probably need to spend some time registering accounts on a number of different websites. More details on API key setup are available [here](http://tyarkoni.github.io/pliers/installation.html#api-keys).

## Documentation

The [Pliers Documentation](http://tyarkoni.github.io/pliers/) contains extensive documentation, including a comprehensive user guide as well as a complete [API Reference](http://tyarkoni.github.io/pliers/reference.html).

