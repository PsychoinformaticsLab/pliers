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
