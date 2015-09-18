import sys
from setuptools import setup
from featurex.version import __version__

if len(set(('test', 'easy_install')).intersection(sys.argv)) > 0:
    import setuptools

extra_setuptools_args = {}
if 'setuptools' in sys.modules:
    extra_setuptools_args = dict(
        tests_require=['nose'],
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
    install_requires=['numpy', 'scipy', 'pandas', 'six', 'python-magic'],
    packages=["featurex"],
    package_data={'featurex': ['data/*'],
                  'featurex.tests': ['data/*']
                  },
    # download_url='https://github.com/tyarkoni/featurex/archive/%s.tar.gz' %
    # __version__,
    **extra_setuptools_args
)
