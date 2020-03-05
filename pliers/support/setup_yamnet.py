import os
from io import BytesIO
from zipfile import ZipFile
from urllib import request
from pathlib import Path
import sys
import runpy

YAMNET_PATH = Path.home() / 'pliers_data'

def setup_yamnet(download_dir=None):
    if download_dir is not None:
        download_dir = Path(download_dir)
    else:
        download_dir = YAMNET_PATH
    
    model_dir = download_dir / 'models-master' / 'research' / 'audioset' / 'yamnet'
    repo_url = 'https://github.com/tensorflow/models/archive/master.zip'
    model_url = 'https://storage.googleapis.com/audioset/yamnet.h5'
    model_filename =  model_dir / model_url.split('/')[-1]

    if not model_filename.exists():
        download_dir.mkdir(exist_ok=True)
        with request.urlopen(repo_url) as z:
            with ZipFile(BytesIO(z.read())) as zfile:
                zfile.extractall(download_dir)
        size = model_dir.stat().st_size
        print(f'Model repository downloaded at {str(download_dir)} '
              f', size: {size} bytes\n')
  
        request.urlretrieve(model_url, str(model_filename))
        print(f'Succesfully downloaded model file.\n')

    test_path = model_dir / 'yamnet_test.py'
    sys.path.insert(0, str(model_dir))
    os.chdir(model_dir)
    runpy.run_path(str(test_path), run_name='__main__')

if __name__ == '__main__':
    setup_yamnet()
