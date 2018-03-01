.. include:: _includes/_replacements.rst

Installing pliers
=================

The easiest way to install pliers is with pip. For the latest stable release:

::

    pip install pliers

Or, if you want to work on the bleeding edge:

::

    pip install pliers git+https://github.com/tyarkoni/pliers.git

Dependencies
------------

By default, installing pliers with pip will only install third-party libraries that are essential for pliers to function properly. These libraries are listed in requirements.txt. However, because pliers provides interfaces to a large number of feature extraction tools, there are literally dozens of other optional dependencies that may be required depending on 
what kinds of features you plan to extract (see optional-dependencies.txt). To be on the safe side, you can install all of the optional dependencies with pip:

pip install -r optional-dependencies.txt

Note, however, that some of these Python dependencies have their own (possibly platform-dependent) requirements. Most notably, python-magic requires libmagic (see `here <https://github.com/ahupp/python-magic#dependencies>`_ for installation instructions), and without this, you'll be relegated to loading all your stims explicitly rather than passing in filenames (i.e., :py:`stim = VideoStim('my_video.mp4'`) will work fine, but passing 'my_video.mp4' directly to an |Extractor| will not). Additionally, the Python OpenCV bindings require `OpenCV3 <http://opencv.org/>`_ (which can be a bit more challenging to install)--but relatively few of the feature extractors in pliers currently depend on OpenCV, so you may not need to bother with this. Similarly, the |TesseractConverter| requires the tesseract OCR library, but no other |Transformer| does, so unless you're planning to capture text from images, you're probably safe.

API Keys
--------
While installing pliers itself is usually straightforward, setting up some of the web-based feature extraction APIs that pliers interfaces with can take a bit more effort. For example, pliers includes support for face and object recognition via Google's Cloud Vision API, and enables conversion of audio files to text transcripts via several different speech-to-text services. While some of these APIs are free to use (and virtually all provide a limited number of free monthly calls), they all require each user to register for their own API credentials. This means that, in order to get the most out of pliers, you'll probably need to spend some time registering accounts on a number of different websites. The following table lists all of the APIs supported by pliers at the moment, along with registration URLs:

+---------------------------------------+-----------------------------------------------------------------------------------------------------+--------------------------------------+--------------------------------+---------------------------------------+
| Transformer class                     | Web service                                                                                         | Environment variable(s)              | Variable description           | Example values                        |
+=======================================+=====================================================================================================+======================================+================================+=======================================+
| WitTranscriptionConverter             | `Wit.ai speech-to-text API <http://wit.ai>`__                                                       | WIT\_AI\_API\_KEY                    | Server Access Token            | A27C1HPZBEDVLW1T1IJAR3L2Q2DA6K3D      |
+---------------------------------------+-----------------------------------------------------------------------------------------------------+--------------------------------------+--------------------------------+---------------------------------------+
| IBMSpeechAPIConverter                 | `IBM Watson speech-to-text API <https://www.ibm.com/watson/developercloud/speech-to-text.html>`__   | IBM\_USERNAME                        | API username and password      | 98452-bvc42-fd-42221-cv21 (username*) |
|                                       |                                                                                                     | IBM\_PASSWORD                        |                                | FJ14fns21N1f (password)               |
+---------------------------------------+-----------------------------------------------------------------------------------------------------+--------------------------------------+--------------------------------+---------------------------------------+
| GoogleSpeechAPIConverter              | `Google Cloud Speech API <https://cloud.google.com/speech/>`__                                      | GOOGLE\_APPLICATION\_CREDENTIALS     | path to .json discovery file   | path/to/credentials.json              |
+---------------------------------------+-----------------------------------------------------------------------------------------------------+--------------------------------------+--------------------------------+---------------------------------------+
| GoogleVisionAPITextConverter          | `Google Cloud Vision API <https://cloud.google.com/vision/>`__                                      | GOOGLE\_APPLICATION\_CREDENTIALS     | path to .json discovery file   | path/to/credentials.json              |
+---------------------------------------+-----------------------------------------------------------------------------------------------------+--------------------------------------+--------------------------------+---------------------------------------+
| GoogleVisionAPIFaceExtractor          | `Google Cloud Vision API <https://cloud.google.com/vision/>`__                                      | GOOGLE\_APPLICATION\_CREDENTIALS     | path to .json discovery file   | path/to/credentials.json              |
+---------------------------------------+-----------------------------------------------------------------------------------------------------+--------------------------------------+--------------------------------+---------------------------------------+
| GoogleVisionAPILabelExtractor         | `Google Cloud Vision API <https://cloud.google.com/vision/>`__                                      | GOOGLE\_APPLICATION\_CREDENTIALS     | path to .json discovery file   | path/to/credentials.json              |
+---------------------------------------+-----------------------------------------------------------------------------------------------------+--------------------------------------+--------------------------------+---------------------------------------+
| GoogleVisionAPIPropertyExtractor      | `Google Cloud Vision API <https://cloud.google.com/vision/>`__                                      | GOOGLE\_APPLICATION\_CREDENTIALS     | path to .json discovery file   | path/to/credentials.json              |
+---------------------------------------+-----------------------------------------------------------------------------------------------------+--------------------------------------+--------------------------------+---------------------------------------+
| GoogleVisionAPIWebEntitiesExtractor   | `Google Cloud Vision API <https://cloud.google.com/vision/>`__                                      | GOOGLE\_APPLICATION\_CREDENTIALS     | path to .json discovery file   | path/to/credentials.json              |
+---------------------------------------+-----------------------------------------------------------------------------------------------------+--------------------------------------+--------------------------------+---------------------------------------+
| MicrosoftAPITextConverter             | `Microsoft Computer Vision API < https://azure.microsoft.com/try/cognitive-services/my-apis/>`__    | MICROSOFT\_VISION\_SUBSCRIPTION\_KEY | API key                        | 152b067184e2ae03711e6439de124c27      |
|                                       |                                                                                                     | MICROSOFT\_SUBSCRIPTION\_LOCATION    | API registered region          | westus                                |
+---------------------------------------+-----------------------------------------------------------------------------------------------------+--------------------------------------+--------------------------------+---------------------------------------+
| MicrosoftVisionAPIExtractor           | `Microsoft Computer Vision API < https://azure.microsoft.com/try/cognitive-services/my-apis/>`__    | MICROSOFT\_VISION\_SUBSCRIPTION\_KEY | API key                        | 152b067184e2ae03711e6439de124c27      |
| (and subclasses)                      |                                                                                                     | MICROSOFT\_SUBSCRIPTION\_LOCATION    | API registered region          | westus                                |
+---------------------------------------+-----------------------------------------------------------------------------------------------------+--------------------------------------+--------------------------------+---------------------------------------+
| MicrosoftAPIFaceExtractor             | `Microsoft Face API < https://azure.microsoft.com/try/cognitive-services/my-apis/>`__               | MICROSOFT\_FACE\_SUBSCRIPTION\_KEY   | API key                        | 152b067184e2ae03711e6439de124c27      |
| (and subclasses)                      |                                                                                                     | MICROSOFT\_SUBSCRIPTION\_LOCATION    | API registered region          | westus                                |
+---------------------------------------+-----------------------------------------------------------------------------------------------------+--------------------------------------+--------------------------------+---------------------------------------+
| IndicoAPIExtractor                    | `Indico.io API <https://indico.io>`__                                                               | INDICO\_APP\_KEY                     | API key                        | 45f9f8a56e4194d3dce858db1e5c3ae4      |
+---------------------------------------+-----------------------------------------------------------------------------------------------------+--------------------------------------+--------------------------------+---------------------------------------+
| ClarifaiAPIExtractor                  | `Clarifai image recognition API <https://clarifai.com>`__                                           | CLARIFAI\_API\_KEY                   | API key                        | 168ed02e137459ead66c3a661be7b784      |
+---------------------------------------+-----------------------------------------------------------------------------------------------------+--------------------------------------+--------------------------------+---------------------------------------+

* - note that this is not the plaintext e-mail or username for your IBM services account

Once you've obtained API keys for the services you intend to use, there are two ways to get pliers to recognize and use your credentials. First, each API-based Transformer can be passed the necessary values (or a path to a file containing those values) as arguments at initialization. For example:

::

    from pliers.extractors import ClarifaiAPIExtractor
    ext = ClarifaiAPIExtractor(app_id='my_clarifai_app_id',
                               app_secret='my_clarifai_app_secret')

Alternatively, you can store the appropriate values as environment variables, in which case you can initialize a Transformer without any arguments. This latter approach is generally preferred, as it doesn't require you to hardcode potentially sensitive values into your code. The mandatory environment variable names for each service are listed in the table above.

::

    from pliers.extractors import GoogleVisionAPIFaceExtractor
    # Works fine if GOOGLE_APPLICATION_CREDENTIALS is set in the environment
    ext = GoogleVisionAPIFaceExtractor()
