from setuptools import setup, find_packages
import os

def read_requirements(req):
    with open(req) as f:
        return f.read().splitlines()

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
    url='http://github.com/psychoinformaticslab/pliers',
    install_requires=read_requirements(os.path.join(thispath, "requirements.txt")),
    packages=find_packages(exclude=['pliers/tests']),
    license='MIT',
    package_data={'pliers': ['datasets/*'],
                  'pliers.tests': ['data/*/*'],
                  },
    zip_safe=False,
    download_url='https://github.com/psychoinformaticslab/pliers/archive/%s.tar.gz' %
    __version__,
    extras_require={
        'all': read_requirements(os.path.join(thispath, "optional-dependencies.txt")),
        'docs': read_requirements(os.path.join(thispath, "docs", "requirements.txt")),
        'tests': [
            'pytest',
            'coveralls',
            'pytest-cov',
            'pytest-forked',
            'pytest-xdist',
        ]
    },
    python_requires='>=3.6',
)
