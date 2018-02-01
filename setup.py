from os.path import dirname, join as opj
from setuptools import setup, find_packages


def get_version():
    """Load version from version.py without entailing any imports
    """
    # This might entail lots of imports which might not yet be available
    # so let's do ad-hoc parsing of the version.py
    with open(opj(dirname(__file__), 'pliers', 'version.py')) as f:
        version_lines = list(filter(lambda x: x.startswith('__version__'), f))
    assert (len(version_lines) == 1)
    return version_lines[0].split('=')[1].strip(" '\"\t\n")


extra_setuptools_args = dict(
    tests_require=['pytest']
)

__version__ = get_version()

setup(
    name="pliers",
    version=__version__,
    description="Multimodal feature extraction in Python",
    maintainer='Tal Yarkoni',
    maintainer_email='tyarkoni@gmail.com',
    url='http://github.com/tyarkoni/pliers',
    install_requires=['numpy', 'scipy', 'moviepy', 'pandas', 'six',
                      'pillow', 'python-magic', 'requests', 'nltk'],
    packages=find_packages(exclude=['pliers/tests']),
    license='MIT',
    package_data={'pliers': ['datasets/*'],
                  'pliers.tests': ['data/*/*']
                  },
    zip_safe=False,
    download_url='https://github.com/tyarkoni/pliers/archive/%s.tar.gz' %
        __version__,
    **extra_setuptools_args
)
