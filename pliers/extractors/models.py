''' Extractor classes based on pre-trained models. '''

import numpy as np
import pandas as pd

from pliers.extractors.image import ImageExtractor
from pliers.extractors.base import Extractor, ExtractorResult
from pliers.filters.image import ImageResizingFilter
from pliers.stimuli import ImageStim, TextStim
from pliers.stimuli.base import Stim
from pliers.support.exceptions import MissingDependencyError
from pliers.utils import (attempt_to_import, verify_dependencies,
                         listify)

import logging

tf = attempt_to_import('tensorflow')
hub = attempt_to_import('tensorflow_hub')
attempt_to_import('tensorflow.keras')
attempt_to_import('tensorflow_text')


class TFHubExtractor(Extractor):

    ''' A generic class for Tensorflow Hub extractors 
    Args:
        url_or_path (str): url or path to TFHub model. You can
            browse models at https://tfhub.dev/.
        features (optional): list of labels (for classification) 
            or other feature names. The number of items must 
            match the number of features in the output. For example,
            if a classification model with 1000 output classes is passed 
            (e.g. EfficientNet B6, 
            see https://tfhub.dev/tensorflow/efficientnet/b6/classification/1), 
            this must be a list containing 1000 items. If a text encoder 
            outputting 768-dimensional encoding is passed (e.g. base BERT),
            this must be a list containing 768 items. Each dimension in the 
            model output will be returned as a separate feature in the 
            ExtractorResult.
            Alternatively, the model output can be packed into a single 
            feature (i.e. a vector) by passing a single-element list 
            (e.g. ['encoding']) or a string. Along the lines of 
            the previous examples, if a single feature name is 
            passed here (e.g. if features=['encoding']) for a TFHub model 
            that outputs a 768-dimensional encoding, the extractor will 
            return only one feature named 'encoding', which contains the 
            encoding vector as a 1-d array wrapped in a list.
            If no value is passed, the extractor will automatically 
            compute the number of features in the model output 
            and return an equal number of features in pliers, labeling
            each feature with a generic prefix + its positional index 
            in the model output (feature_0, feature_1, ... ,feature_n).
        transform_out (optional): function to transform model 
            output for compatibility with extractor result
        transform_inp (optional): function to transform Stim.data 
            for compatibility with model input format
        keras_args (dict): arguments to hub.KerasLayer call
    '''

    _log_attributes = ('url_or_path', 'features', 'transform_out', 'keras_args')
    _input_type = Stim

    def __init__(self, url_or_path, features=None,
                 transform_out=None, transform_inp=None,
                 keras_args=None):
        verify_dependencies(['tensorflow_hub'])
        if keras_args is None:
            keras_args = {}
        self.keras_args = keras_args
        self.model = hub.KerasLayer(url_or_path, **keras_args)
        self.url_or_path = url_or_path
        self.features = features
        self.transform_out = transform_out
        self.transform_inp = transform_inp
        super().__init__()

    def get_feature_names(self, out):
        if self.features:
            return listify(self.features)
        else:
            return ['feature_' + str(i) 
                    for i in range(out.shape[-1])]
    
    def _preprocess(self, stim):
        if self.transform_inp:
            return self.transform_inp(stim.data)
        else:
            if type(stim) == TextStim:
                return listify(stim.data)
            else:
                return stim.data

    def _postprocess(self, out):
        if self.transform_out:
            out = self.transform_out(out)
        return out.numpy().squeeze()
        
    def _extract(self, stim):
        inp = self._preprocess(stim)
        out = self.model(inp)
        out = self._postprocess(out)
        features = self.get_feature_names(out)
        return ExtractorResult(listify(out), stim, self, 
                               features=features)
    
    def _to_df(self, result):
        if len(result.features) == 1:
            data = [result._data]
        else:
            data = np.array(result._data)
            if len(data.shape) > 2:
                data = data.squeeze()
        res_df = pd.DataFrame(data, columns=result.features)
        return res_df


class TFHubImageExtractor(TFHubExtractor):

    ''' TFHub Extractor class for image models
    Args:
        url_or_path (str): url or path to TFHub model
        features (optional): list of labels (for classification) 
            or other feature names. If not specified, returns 
            numbered features (feature_0, feature_1, ... ,feature_n)
        keras_args (dict): arguments to hub.KerasLayer call
    '''

    _input_type = ImageStim
    _log_attributes = ('url_or_path', 'features', 'keras_args')

    def __init__(self, 
                 url_or_path, 
                 features=None,
                 input_dtype=tf.float32,
                 keras_args=None):
        
        self.input_dtype = input_dtype
        if keras_args is None:
            keras_args = {}
        self.keras_args = keras_args

        logging.warning('Some models may require specific input shapes.'
                        ' Incompatible shapes may raise errors'
                        ' at extraction. If needed, you can reshape'
                        ' your input image using ImageResizingFilter, '
                        ' and rescale using ImageRescalingFilter')
        super().__init__(url_or_path, features, keras_args=keras_args)

    def _preprocess(self, stim):
        x = tf.convert_to_tensor(stim.data, dtype=self.input_dtype)
        x = tf.expand_dims(x, axis=0)
        return x


class TFHubTextExtractor(TFHubExtractor):

    ''' TFHub extractor class for text models
    Args:
        url_or_path (str): url or path to TFHub model. You can
            browse models at https://tfhub.dev/.
        features (optional): list of labels or other feature names. 
            The number of items must  match the number of features 
            in the model output. For example, if a text encoder 
            outputting 768-dimensional encoding is passed 
            (e.g. base BERT), this must be a list containing 768 items. 
            Each dimension in the model output will be returned as a 
            separate feature in the ExtractorResult.
            Alternatively, the model output can be packed into a single 
            feature (i.e. a vector) by passing a single-element list 
            (e.g. ['encoding']) or a string. If no value is passed, 
            the extractor will automatically compute the number of 
            features in the model output and return an equal number 
            of features in pliers, labeling each feature with a 
            generic prefix + its positional index in the model 
            output (feature_0, feature_1, ... ,feature_n).
        output_key (str): key to desired embedding in output 
            dictionary (see documentation at 
            https://www.tensorflow.org/hub/common_saved_model_apis/text).
            Set to None is the output is not a dictionary.
        preprocessor_url_or_path (str): if the model requires 
            preprocessing through another TFHub model, specifies the 
            url or path to the preprocessing module. Information on 
            required preprocessing and appropriate models is generally
            available on the TFHub model webpage
        preprocessor_kwargs (dict): dictionary or named arguments
            for preprocessor model hub.KerasLayer call
    '''

    _input_type = TextStim 

    def __init__(self,
                 url_or_path, 
                 features=None,
                 output_key='default',
                 preprocessor_url_or_path=None, 
                 preprocessor_kwargs=None,
                 keras_args=None,
                 **kwargs):
        super().__init__(url_or_path, features, 
                         keras_args=keras_args, 
                         **kwargs)
        self.output_key = output_key
        self.preprocessor_url_or_path=preprocessor_url_or_path
        self.preprocessor_kwargs = preprocessor_kwargs
        try:
            verify_dependencies(['tensorflow_text'])
        except MissingDependencyError:
            logging.warning('Some TFHub text models require TensorFlow Text '
                            '(see https://www.tensorflow.org/tutorials/tensorflow_text/intro),'
                            ' which is not installed.'
                            ' Missing dependency errors may arise.')

    def _preprocess(self, stim):
        x = listify(stim.data)
        if self.preprocessor_url_or_path:
            preprocessor = hub.KerasLayer(self.preprocessor_url_or_path,
                                          self.preprocessor_kwargs)
            x = preprocessor(x)
        return x
    
    def _postprocess(self, out):
        if not self.output_key:
            return out.numpy().squeeze()
        else:
            try:
                return out[self.output_key].numpy().squeeze()
            except KeyError:
                raise ValueError(f'{self.output_key} is not a valid key.'
                                'Check which keys are available in the output '
                                'embedding dictionary in TFHub docs '
                                '(https://www.tensorflow.org/hub/common_saved_model_apis/text)'
                                f' or at the model URL ({self.url_or_path})')
            except (IndexError, TypeError):
                raise ValueError(f'Model output is not a dictionary. '
                                  'Try initialize the extractor with output_key=None.')


class TensorFlowKerasApplicationExtractor(ImageExtractor):

    ''' Labels objects in images using a pretrained Inception V3 architecture
    implemented in TensorFlow / Keras.

    Images must be RGB and be a certain shape. Different model architectures
    may require different shapes, and images will be resized (with some
    distortion) if the shape of the image is different.

    Args:
        architecture (str): model architecture to use. One of 'vgg16', 'vgg19',
            'resnet50', 'inception_resnetv2', 'inceptionv3', 'xception',
            'densenet121', 'densenet169', 'nasnetlarge', or 'nasnetmobile'.
        weights (str): URL to download pre-trained weights. If None (default),
            uses the pre-trained weights trained on ImageNet used in Keras
            Applications.
        num_predictions (int): Number of top predicted labels to retain for
            each image.
     '''

    _log_attributes = ('architecture', 'weights', 'num_predictions')
    VERSION = '1.0'

    def __init__(self,
                 architecture='inceptionv3',
                 weights=None,
                 num_predictions=5):
        verify_dependencies(['tensorflow'])
        verify_dependencies(['tensorflow.keras'])
        super().__init__()

        # Model name: (model module, model function, required shape).
        apps = tf.keras.applications
        model_mapping = {
            'vgg16': (apps.vgg16, apps.vgg16.VGG16, (224, 224, 3)),
            'vgg19': (apps.vgg19, apps.VGG19, (224, 224, 3)),
            'resnet50': (apps.resnet50, apps.resnet50.ResNet50, (224, 224, 3)),
            'inception_resnetv2': (
                apps.inception_resnet_v2,
                apps.inception_resnet_v2.InceptionResNetV2, (299, 299, 3)),
            'inceptionv3': (
                apps.inception_v3, apps.inception_v3.InceptionV3,
                (299, 299, 3)),
            'xception': (apps.xception, apps.xception.Xception, (299, 299, 3)),
            'densenet121': (
                apps.densenet, apps.densenet.DenseNet121, (224, 224, 3)),
            'densenet169': (
                apps.densenet, apps.densenet.DenseNet169, (224, 224, 3)),
            'densenet201': (
                apps.densenet, apps.densenet.DenseNet201, (224, 224, 3)),
            'nasnetlarge': (
                apps.nasnet, apps.nasnet.NASNetLarge, (331, 331, 3)),
            'nasnetmobile': (
                apps.nasnet, apps.nasnet.NASNetMobile, (224, 224, 3)),
        }
        if weights is None:
            weights = 'imagenet'
        if architecture.lower() not in model_mapping.keys():
            raise ValueError(
                "Unknown architecture '{}'. Available arhitectures are '{}'."
                .format(architecture, "', '".join(model_mapping.keys())))

        self.architecture = architecture.lower()
        self.weights = weights
        self.num_predictions = num_predictions

        # The preprocessing and decoding functions are in the module.
        self._model_module = model_mapping[self.architecture][0]
        # Instantiating the model also downloads the weights to a cache.
        self.model = model_mapping[self.architecture][1](weights=self.weights)
        self._required_shape = model_mapping[self.architecture][2]

    def _extract(self, stim):
        x = stim.data
        if x.ndim != 3:
            raise ValueError(
                "Stim data must have rank 3 but got rank {}".format(x.ndim))
        if x.shape != self._required_shape:
            resizer = ImageResizingFilter(size=self._required_shape[:-1])
            x = resizer.transform(stim).data
        # Add batch dimension.
        x = x[None]
        # Normalize the features.
        x = self._model_module.preprocess_input(x)
        preds = self.model.predict(x, batch_size=1)

        # This produces a nested list. There is one sub-list per sample in the
        # batch, and each sub-list contains `self.num_predictions` tuples with
        # `(ID, human-readable-label, probability)`.
        decoded = self._model_module.decode_predictions(
            preds, top=self.num_predictions)

        # We assume there is only one sample in the batch.
        decoded = decoded[0]
        values = [t[2] for t in decoded]
        features = [t[1] for t in decoded]

        return ExtractorResult([values], stim, self, features=features)
