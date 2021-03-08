from setuptools import setup, find_packages
import os

extra_setuptools_args = dict(
    tests_require=['pytest']
)

thispath, _ = os.path.split(__file__)

ver_file = os.path.join(thispath, 'pliers', 'version.py')

with open(ver_file) as fp:
    exec(fp.read(), globals(), locals())

setup(
    name="pliers",
    version=locals()['__version__'],
    description="Multimodal feature extraction in Python",
    maintainer='Tal Yarkoni',
    maintainer_email='tyarkoni@gmail.com',
    url='http://github.com/tyarkoni/pliers',

    install_requires=[
        'imageio>=2.3',
        'moviepy>=0.2',
        'nltk>=3.0',
        'numpy>=1.13',
        'pandas>=0.24',
        'pillow',
        'psutil',
        'python-magic',
        'requests',
        'scipy>=0.13',
        'tqdm'
    ],
    packages=find_packages(exclude=['pliers/tests']),
    license='MIT',
    package_data={'pliers': ['datasets/*'],
                  'pliers.tests': ['data/*/*']
                  },
    zip_safe=False,
    download_url='https://github.com/tyarkoni/pliers/archive/%s.tar.gz' %
    __version__,
    **extra_setuptools_args,
    extras_require={
        'extractors': [
            'clarifai',
            'duecredit',
            'face_recognition',
            'gensim',
            'google-api-python-client',
            'google-compute-engine',
            'librosa>=0.6.3',
            'numba<=0.48',
            'matplotlib',
            'opencv-python',
            'openpyxl',
            'pathos',
            'pygraphviz',
            'pysrt',
            'pytesseract',
            'python-twitter',
            'rev_ai',
            'scikit-learn',
            'seaborn',
            'spacy',
            'SpeechRecognition>=3.6.0',
            'tensorflow>=2.0.0',
            'torch',
            'transformers',
            'tensorflow-hub',
            'tensorflow_text',
            'xlrd'
        ],
        'docs': [
            'sphinx-rtd-theme',
            'sphinx-gallery',
        ],
        'tests': [
            'coveralls',
            'pytest-cov',
            'pytest-forked',
            'pytest-xdist',
        ]
    },
    python_requires='>=3.6',
)
