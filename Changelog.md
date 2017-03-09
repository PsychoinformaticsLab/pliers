# Changelog

## 0.1.1 (March 9, 2017)
This is a minor release that adds a number of new (minor) features and a bunch of bug fixes. Changes include:
* Added very basic parallelization support
* Added support for generating graph diagrams using PyGraphViz (via Graph.plot())
* Added a new RemoteStim that represents remote, uninitialized resources
* Added a new WordStemmingFilter that stems text using nltk
* Added support for Google Cloud Vision's safe image detection
* IBMSpeechAPIConverter now takes an argument indicating whether to return words or phrases
* Added support for image-based Indico API models
* Add support for multiple results returned by IBM speech converter
* General clean-up, code styling, and docstring addition
* Bug fixes:
    - Update various Transformers to reflect changes to APIs
    - Fixed dataset import bug that prevented PyPI-installed 0.1 from importing properly
    - Update args to numpy functions to ensure numpy 1.12+ compatibility
    - Various other minor bug fixes

## 0.1 (January 17, 2017)
First major release. Nothing before this counts.