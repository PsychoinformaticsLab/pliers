''' Extractor classes based on pre-trained models. '''

import os
import tempfile
import tarfile
import subprocess
import re
import requests
from pliers.extractors.image import ImageExtractor
from pliers.extractors.base import ExtractorResult


class TensorFlowInceptionV3Extractor(ImageExtractor):

    ''' Labels objects in images using a pretrained Inception V3 architecture
     implemented in TensorFlow.

    Args:
        model_dir (str): path to save model file to. If None (default), creates
            and uses a temporary folder.
        data_url (str): URL to download model from. If None (default), uses
            the preset inception model (dated 2015-12-05) used in the
            TensoryFlow tutorials.
        num_predictions (int): Number of top predicted labels to retain for
            each image.
     '''

    _log_attributes = ('model_dir', 'data_url', 'num_predictions')
    VERSION = '1.0'

    def __init__(self, model_dir=None, data_url=None, num_predictions=5):

        super(TensorFlowInceptionV3Extractor, self).__init__()

        if model_dir is None:
            model_dir = os.path.join(tempfile.gettempdir(), 'TFInceptionV3')
        self.model_dir = model_dir

        if data_url is None:
            data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        self.data_url = data_url

        filename = self.data_url.split('/')[-1]
        self.model_file = os.path.join(self.model_dir, filename)
        self.num_predictions = num_predictions

        # Download the inception-v3 model if needed
        if not os.path.exists(self.model_file):
            self._download_pretrained_model()

    def _download_pretrained_model(self):
        # Adapted from def_maybe_download_and_extract() in TF's
        # classify_image.py
        print("Downloading Inception-V3 model from TensorFlow website...")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        filename = os.path.basename(self.model_file)
        if not os.path.exists(self.model_file):
            r = requests.get(self.data_url)
            with open(self.model_file, 'wb') as f:
                f.write(r.content)
            size = os.stat(self.model_file).st_size
            print('\tSuccesfully downloaded', filename, size, 'bytes.')
            tarfile.open(self.model_file, 'r:gz').extractall(self.model_dir)

    def _extract(self, stim):
        from pliers.external import tensorflow as tf
        tf_dir = os.path.dirname(tf.__file__)
        script = os.path.join(tf_dir, 'classify_image.py')

        with stim.get_filename() as filename:
            args = ' --image_file %s --model_dir %s --num_top_prediction %d' % \
                (filename, self.model_dir, self.num_predictions)
            cmd = ('python ' + script + args).split()
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            output, errors = process.communicate()
            hits = output.decode('utf-8').splitlines()[-self.num_predictions:]

        values, features = [], []
        for i, h in enumerate(hits):
            m = re.search('(.*?)\s\(score\s\=\s([0-9\.]+)\)', h.strip())
            extraction = m.groups()
            features.append(extraction[0])
            values.append(float(extraction[1]))

        return ExtractorResult([values], stim, self, features=features)
