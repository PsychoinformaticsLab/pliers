'''
Extractors that interact with external (e.g., deep learning) services.
'''

from featurex.extractors.image import ImageExtractor
from scipy.misc import imsave
from featurex.core import Value
import os

try:
    from metamind.api import (set_api_key, get_api_key,
                              general_image_classifier, ClassificationModel)
except ImportError:
    pass

try:
    from clarifai.client import ClarifaiApi
except ImportError:
    pass


class MetamindAPIExtractor(APIExtractor, ImageExtractor):

    ''' Uses the MetaMind API to extract features with an existing classifier.
    Args:
        api_key (str): A valid key for the MetaMind API. Only needs to be
            passed the first time a MetaMindExtractor is initialized.
        classifier (str, int): The name or ID of the MetaMind classifier to
            use. If None or 'general', defaults to the general image
            classifier. Otherwise, must be an integer ID for the desired
            classifier.
    '''

    def __init__(self, api_key=None, classifier=None):
        ImageExtractor.__init__(self)
        api_key = get_api_key() if api_key is None else api_key
        if api_key is None:
            raise ValueError("A valid MetaMind API key must be passed the "
                             "first time a MetaMind extractor is initialized.")
        set_api_key(api_key, verbose=False)

        # TODO: Can add a lookup dictionary somewhere that has name --> ID
        # translation for commonly used classifiers.
        if classifier is None:
            self.classifier = general_image_classifier
        else:
            self.classifier = ClassificationModel(id=classifier)

    def apply(self, img):
        data = img.data
        temp_file = tempfile.mktemp() + '.png'
        imsave(temp_file, data)
        labels = self.classifier.predict(temp_file, input_type='files')
        os.remove(temp_file)
        time.sleep(1.0)  # Prevents server error somewhat

        return Value(img, self, {'labels': labels})


class ClarifaiAPIExtractor(APIExtractor, ImageExtractor):

    ''' Uses the Clarifai API to extract tags of images.
    Args:
        app_id (str): A valid APP_ID for the Clarifai API. Only needs to be
            passed the first time the extractor is initialized.
        app_secret (str): A valid APP_SECRET for the Clarifai API. 
            Only needs to be passed the first time the extractor is initialized.
        model (str): The name of the Clarifai model to use. 
            If None, defaults to the general image tagger. 
    '''

    def __init__(self, app_id=None, app_secret=None, model=None, select_classes=None):
        ImageExtractor.__init__(self)
        if app_id is None or app_secret is None:
            try:
                app_id = os.environ['CLARIFAI_APP_ID']
                app_secret = os.environ['CLARIFAI_APP_SECRET']
            except KeyError:
                raise ValueError("A valid Clarifai API APP_ID and APP_SECRET"
                                 "must be passed the first time a Clarifai "
                                 "extractor is initialized.")

        self.tagger = ClarifaiApi(app_id=app_id, app_secret=app_secret)
        if not (model is None):
            self.tagger.set_model(model)

        self.select_classes = select_classes

    def apply(self, img):
        data = img.data
        temp_file = tempfile.mktemp() + '.png'
        imsave(temp_file, data)
        tags = self.tagger.tag_images(open(temp_file, 'rb'), select_classes=self.select_classes)
        os.remove(temp_file)

        return Value(img, self, {'tags': tags})
