''' Extractor classes based on pre-trained models. '''

import numpy as np
import pandas as pd

from pliers.extractors.image import ImageExtractor
from pliers.extractors.base import Extractor, ExtractorResult
from pliers.filters.image import ImageResizingFilter
from pliers.stimuli import ImageStim, TextStim, AudioStim
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
        load_type (optional): whether to load the model as a KerasLayer
            or as a SavedModel. If 'keras', loads as a KerasLayer. If
            'saved_model', loads as a SavedModel.
        signature (optional): signature to use when loading a SavedModel.
        features (optional): list of feature names matching output dimensions
            
            For example, for a classification model with 1000 output classes 
            this must be a list containing 1000 items.
            (e.g. EfficientNet B6, 
            https://tfhub.dev/tensorflow/efficientnet/b6/classification/1), 
            
            Alternatively, the model output can be packed into a single 
            feature (i.e. a vector) by passing a single-element list 
            or a string. For example, for a model that outputs a 
            768-dimensional encoding, the value 'encoding' will result
            in a 1-d array wrapped in a list named 'encoding'.

            If no value is passed, the extractor will automatically 
            compute the number of features in the model output 
            and return an equal number of features in pliers, labeling
            each feature with a generic prefix + its positional index 
            in the model output (feature_0, feature_1, ... ,feature_n).

            Note that for saved models, the feature names are inferred
            from the output signature, but can be over-ridden.
        transform_out (optional): function to transform model 
            output for compatibility with extractor result
        transform_inp (optional): function to transform Stim.data 
            for compatibility with model input format
        keras_kwargs (dict): arguments to hub.KerasLayer call
    '''

    _log_attributes = ('url_or_path', 'features', 'transform_out', 'keras_kwargs')
    _input_type = Stim

    def __init__(self, url_or_path, features=None,
                 transform_out=None, transform_inp=None,
                 load_type='keras', signature=None,
                 keras_kwargs=None):
        verify_dependencies(['tensorflow_hub'])
        if keras_kwargs is None:
            keras_kwargs = {}
        self.keras_kwargs = keras_kwargs
        self.load_type = load_type

        if load_type == 'keras':
            self.model = hub.KerasLayer(url_or_path, **keras_kwargs)
        elif load_type == 'saved_model':
            model = hub.load(url_or_path)
            if signature is None:
                signature = list(model.signatures.keys())[0]
            self.model = model.signatures[signature]

        self.url_or_path = url_or_path
        self.features = features
        self.transform_out = transform_out
        self.transform_inp = transform_inp
        super().__init__()

    def get_feature_names(self, out):
        if self.features:
            return listify(self.features)
        else:
            if isinstance(out, dict):
                return list(out.keys())
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
        if not isinstance(out, np.ndarray):
            out = out.numpy().squeeze()
        return out

    def _get_timing(self, out, stim):
        """ Returns the timing of the output. 
        Args:
            out: output of the model
            stim: input stimulus

        Returns:
            onsets: onsets of the output
            durations: durations of the output        
        """
        return None, None
        
    def _extract(self, stim):
        inp = self._preprocess(stim)
        out = self.model(inp)

        features = self.get_feature_names(out)

        if self.load_type == 'saved_model':
            if isinstance(out, dict):
                out = np.vstack(out.values()).T

        out = self._postprocess(out)
                
        onsets, durations = self._get_timing(out, stim)

        return ExtractorResult(listify(out), stim, self, 
                               onsets=onsets, durations=durations,
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

    ''' TFHub Extractor class for image models.

    Note that some models may require specific input shapes.'
    You can reshape inputs using filters, such as ImageResizingFilter.
    ImageRescaleFilter.

    Args:
        url_or_path (str): url or path to TFHub model
        input_dtype (optional): dtype of input data. Defaults to tf.float32
    '''

    _input_type = ImageStim

    def __init__(self, 
                 url_or_path, 
                 input_dtype=None,
                 **kwargs):
        
        self.input_dtype = input_dtype if input_dtype else tf.float32

        super().__init__(url_or_path, **kwargs)

    def _preprocess(self, stim):
        x = tf.convert_to_tensor(stim.data, dtype=self.input_dtype)
        x = tf.expand_dims(x, axis=0)
        return x

class TFHubAudioExtractor(TFHubExtractor):

    ''' TFHub Extractor class for audio models.

    Note that some models may require a specific sampling frequency.'
    You can resample inputs using AudioResamplingFilter.

    Args:
        url_or_path (str): url or path to TFHub model
        input_dtype (optional): dtype of input data. Defaults to tf.float32
    '''

    _input_type = AudioStim

    def __init__(self, 
                 url_or_path, 
                 input_dtype=None,
                 **kwargs):
        
        self.input_dtype = input_dtype if input_dtype else tf.float32

        super().__init__(url_or_path, **kwargs)

    def _preprocess(self, stim):
        x = tf.convert_to_tensor(stim.data, dtype=self.input_dtype)
        return x

    def _get_timing(self, out, stim):
        """ Returns the timing of the output. 
        Args:
            out: output of the model
            stim: input stimulus

        Returns:
            onsets: onsets of the output
            durations: durations of the output        
        """

        durations = [stim.duration / out.shape[0]] * out.shape[0]
        onsets = np.arange(0, stim.duration, durations[0]).tolist()

        return onsets, durations

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
                 keras_kwargs=None,
                 **kwargs):
        super().__init__(url_or_path, features, 
                         keras_kwargs=keras_kwargs, 
                         **kwargs)
        self.output_key = output_key
        self.preprocessor_url_or_path = preprocessor_url_or_path
        self.preprocessor_kwargs = preprocessor_kwargs if preprocessor_kwargs else {}
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
            preprocessor = hub.KerasLayer(
                self.preprocessor_url_or_path, **self.preprocessor_kwargs)
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
