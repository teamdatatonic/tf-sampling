from setuptools import find_packages
from setuptools import setup

# package meta-data
NAME = 'sampling'
DESCRIPTION = 'Effective sampling methods within TensorFlow input functions.'
URL = 'https://github.com/teamdatatonic/tf-sampling.git'
AUTHOR = 'Will Fletcher and Laxmi Prajapat'
REQUIRES_PYTHON = '>=3.5.0'
VERSION = 0.1

# module package requirements
REQUIRED = [
    'namedtupled>=0.3.3', 'docopt>=0.6.2', 'numpy>=1.14.5', 'gast==0.2.2',
    'scipy>=1.1.0', 'pytest>=5.2.0', 'pytest-cov>=2.8.1',
    'tensorflow==2.11.1', 'attrdict>=2.0.0', 'fastavro>=0.21.17',
    'protobuf>=1.7.0', 'google-api-python-client>=1.7.6',
    'google-cloud-storage>=1.13.0', 'google-cloud-bigquery>=1.7.0'
]

with open('./README.md') as f:
    README = '\n' + f.read()

packages = find_packages()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=README,
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=packages,
    install_requires=REQUIRED,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: MIT License",
    ],
)
