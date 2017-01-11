from pliers.stimuli.image import ImageStim
from pliers.extractors.base import Extractor, ExtractorResult
import numpy as np
import os
import tempfile
import sys
import urllib
import tarfile
from scipy.misc import imsave
import subprocess
import re


class TensorFlowInceptionV3Extractor(Extractor):

    _input_type = ImageStim

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
        # Adapted from def_maybe_download_and_extract() in TF's classify_image.py
        print("Downloading Inception-V3 model from TensorFlow website...")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        filename = os.path.basename(self.model_file)
        if not os.path.exists(self.model_file):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                    float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
            self.model_file, _ = urllib.request.urlretrieve(self.data_url,
                                                self.model_file, _progress)
            statinfo = os.stat(self.model_file)
            print('\tSuccesfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(self.model_file, 'r:gz').extractall(self.model_dir)

    def _extract(self, stim):
        import tensorflow as tf
        tf_dir = os.path.dirname(tf.__file__)
        script = os.path.join(tf_dir, 'models', 'image', 'imagenet', 'classify_image.py')
        
        if stim.filename is None:
            img_file = tempfile.mktemp() + '.jpg'
            imsave(img_file, stim.data)
            use_tmp = True
        else:
            img_file = stim.filename
            use_tmp = False

        args = ' --image_file %s --model_dir %s --num_top_prediction %d' % \
                (img_file, self.model_dir, self.num_predictions)
        cmd = ('python ' + script + args).split()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output, errors = process.communicate()
        hits = output.decode('utf-8').splitlines()[-self.num_predictions:]
        
        values, features = [], []
        for i, h in enumerate(hits):
            m = re.search('(.*?)\s\(score\s\=\s([0-9\.]+)\)', h.strip())
            values.extend(m.groups())
            ind = i + 1
            features.extend(['label_%d' % ind, 'score_%d' % ind])

        if use_tmp:
            os.remove(img_file)

        return ExtractorResult([values], stim, self, features=features)
