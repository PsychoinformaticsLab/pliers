''' Extractor classes based on pre-trained models. '''

import os
import tempfile
import tarfile
import subprocess
import re
import requests
from pliers.extractors.image import ImageExtractor
from pliers.extractors.base import ExtractorResult


def _resize_image(image, shape):
    import numpy as np
    from PIL import Image
    return np.array(
        Image.fromarray(image).resize(shape, resample=Image.BILINEAR))


class TensorFlowKerasInceptionV3Extractor(ImageExtractor):

    ''' Labels objects in images using a pretrained Inception V3 architecture
     implemented in TensorFlow / Keras.

    Args:
        weights (str): URL to download pre-trained weights. If None (default),
            uses the pre-trained Inception V3 model (dated 2017-03-10) used in
            Keras Applications.
        num_predictions (int): Number of top predicted labels to retain for
            each image.
     '''

    _log_attributes = ('weights', 'num_predictions')
    VERSION = '1.0'

    def __init__(self, weights=None, num_predictions=5):
        super(TensorFlowKerasInceptionV3Extractor, self).__init__()
        if weights is None:
            weights = 'imagenet'
        self.weights = weights
        self.num_predictions = num_predictions

        import tensorflow as tf
        # Instantiating the model also downloads the weights to a cache.
        self.model = tf.keras.applications.inception_v3.InceptionV3(
            weights=self.weights)

    def _extract(self, stim):
        import tensorflow as tf

        required_shape = (299, 299, 3)
        x = stim.data
        if x.ndim != 3:
            raise ValueError("Stim must have rank 3.")
        if x.shape != required_shape:
            x = _resize_image(x, required_shape[:-1])
        x = tf.keras.applications.inception_v3.preprocess_input(x)
        x = x[None]  # Add batch dimension.
        preds = self.model.predict(x, batch_size=1)

        # This produces a nested list. There is one sub-list per sample in the
        # batch, and each sub-list contains `self.num_predictions` tuples with
        # `(ID, human-readable-label, probability)`.
        decoded = tf.keras.applications.inception_v3.decode_predictions(
            preds, top=self.num_predictions)

        # We assume there is only one sample.
        decoded = decoded[0]
        values = [t[2] for t in decoded]
        features = [t[1] for t in decoded]

        return ExtractorResult([values], stim, self, features=features)
