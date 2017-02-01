from pliers.version import __version__
from setuptools import setup, find_packages

extra_setuptools_args = dict(
    tests_require=['pytest']
)

setup(
    name="pliers",
    version=__version__,
    description="Multimodal feature extraction in Python",
    maintainer='Tal Yarkoni',
    maintainer_email='tyarkoni@gmail.com',
    url='http://github.com/tyarkoni/pliers',
    install_requires=['numpy<=1.12.0', 'scipy', 'moviepy', 'pandas', 'six',
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
