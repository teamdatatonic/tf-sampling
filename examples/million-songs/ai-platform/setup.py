from setuptools import find_packages
from setuptools import setup

# Package meta-data.
NAME = 'trainer'
DESCRIPTION = 'Million Songs - AI Platform.'
REQUIRES_PYTHON = '>=3.5.0'
VERSION = 0.1

# What packages are required for this module to be executed?
REQUIRED = [
    'namedtupled>=0.3.3', 'docopt>=0.6.2', 'numpy>=1.14.5', 'scipy>=1.1.0',
    'tensorflow==1.15.4', 'attrdict>=2.0.0', 'fastavro>=0.21.17',
    'protobuf>=1.7.0', 'google-api-python-client>=1.7.6',
    'google-cloud-storage>=1.13.0', 'google-cloud-bigquery>=1.7.0'
]

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      python_requires=REQUIRES_PYTHON,
      packages=find_packages(),
      install_requires=REQUIRED,
      include_package_data=True)
