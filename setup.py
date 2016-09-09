from featurex.version import __version__
from setuptools import setup, find_packages

extra_setuptools_args = dict(
    tests_require=['pytest']
)

setup(
    name="featurex",
    version=__version__,
    description="Multimodal feature extraction in Python",
    maintainer='Tal Yarkoni',
    maintainer_email='tyarkoni@gmail.com',
    url='http://github.com/tyarkoni/featurex',
    install_requires=['numpy', 'scipy', 'moviepy', 'pandas', 'six', 'pillow',
                      'python-magic', 'requests', 'nltk'],
    packages=find_packages(exclude=['featurex/tests']),
    license='MIT',
    package_data={'featurex': ['data/*'],
                  'featurex.tests': ['data/*/*']
                  },
    download_url='https://github.com/tyarkoni/featurex/archive/%s.tar.gz' %
    __version__,
    **extra_setuptools_args
)
