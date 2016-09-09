# Changelog

## 0.0.1 (September 9, 2016)
First official release. New features (since 0.0.1a) include:
* New extractors for:
    * Google Cloud APIs
    * Indico API (thanks to @ejolly)
    * Movie subtitle (SRT) extraction (thanks to @tsalo)
    * Mean amplitude extraction from audio (thanks to @tsalo)
    * Image saliency detector (thanks to @shabtastic and @ljchang)
* Added DerivedVideoStim class
* Switched testing to py.test and removed all UnitTest assertions
* Dropped cv2 as a Stim-loading requirement in favor of moviepy and scipy
* Improved test coverage and better travis-ci/coveralls support
* Updated requirements and setup; consolidated optional dependencies in separate file
* Fixed various minor bugs