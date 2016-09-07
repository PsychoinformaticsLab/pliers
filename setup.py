from featurex.version import __version__
from setuptools import setup, find_packages

extra_setuptools_args = dict(
    tests_require=['pytest'],
    test_suite='nose.collector',
    extras_require=dict(
        test='nose>=0.10.1')
)

setup(
    name="featurex",
    version=__version__,
    description="Multimodal feature extraction in Python",
    author='Tal Yarkoni',
    author_email='tyarkoni@gmail.com',
    url='http://github.com/tyarkoni/featurex',
    install_requires=['numpy', 'scipy', 'pandas', 'six', 'python-magic', 'requests'],
    packages=find_packages(),
    package_data={'featurex': ['data/*'],
                  'featurex.tests': ['data/*/*']
                  },
    download_url='https://github.com/tyarkoni/featurex/archive/%s.tar.gz' %
    __version__,
    **extra_setuptools_args
)
