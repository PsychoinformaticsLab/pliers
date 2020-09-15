import os
from io import BytesIO
import json
from zipfile import ZipFile
from urllib import request
from pathlib import Path
import sys
import runpy
import shutil

PLIERS_DATA_PATH = Path.home() / 'pliers_data' 
YAMNET_PATH = PLIERS_DATA_PATH / 'yamnet'


def setup_yamnet():
    # get the most recent released version of tensorflow/models
    release_api_url = "https://api.github.com/repos/tensorflow/models/releases/latest"
    with request.urlopen(release_api_url) as release_url:
        tf_models_dict = json.loads(release_url.read())
        tf_models_version = tf_models_dict['tag_name'].lstrip('v')

    repo_url = f'https://github.com/tensorflow/models/archive/v{tf_models_version}.zip'
    model_url = 'https://storage.googleapis.com/audioset/yamnet.h5'

    tmp_dir = PLIERS_DATA_PATH / 'yamnet_tmp'
    tmp_yamnet_dir = tmp_dir / f'models-{tf_models_version}' / 'research' / 'audioset' / 'yamnet'
    model_filename =  YAMNET_PATH / model_url.split('/')[-1]
    
    if not model_filename.exists():
        PLIERS_DATA_PATH.mkdir(exist_ok=True)
        with request.urlopen(repo_url) as z:
            print('Downloading model repository...\n')
            with ZipFile(BytesIO(z.read())) as zfile:
                zfile.extractall(str(tmp_dir))
        shutil.move(str(tmp_yamnet_dir), str(PLIERS_DATA_PATH))
        shutil.rmtree(str(tmp_dir))
        size = YAMNET_PATH.stat().st_size
        print(f'Model repository downloaded at {str(YAMNET_PATH)} '
              f', size: {size} bytes\n')

        request.urlretrieve(model_url, str(model_filename))
        print(f'Model file downloaded.\n')

    print(YAMNET_PATH)
    test_path = YAMNET_PATH / 'yamnet_test.py'
    sys.path.insert(0, str(YAMNET_PATH))
    os.chdir(YAMNET_PATH)
    runpy.run_path(str(test_path), run_name='__main__')

if __name__ == '__main__':
    setup_yamnet()
