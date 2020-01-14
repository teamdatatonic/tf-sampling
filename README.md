# TensorFlow Sampling

> *Effective sampling methods within TensorFlow input functions.*

**Authors:**
* William Fletcher
* Laxmi Prajapat

## Table of Contents
* [About the Project](#about-the-project)
	* [Built With](#built-with)
	* [Key Features](#key-features)
		* [Sampling Techniques](#sampling-techniques)
		* [Real-World Examples](#real-world-examples)
* [Getting Started](#getting-started)
	* [Installation](#installation)
* [Usage](#usage)
	* [Run Tests](#run-tests)
* [Contributing](#contributing)
* [Licensing](#licensing)

## About the Project
> *A collection of sampling techniques and real-world examples applied to training / testing data directly inside the input function using the tf.data API.*

### Built With
* [Tensorflow](https://www.tensorflow.org/) (1.14)

## Key Features

#### Sampling Techniques
There are two sampling methods presented:
* [Downsampling with weight](sampling/io/downsample.py)
* [Negative sampling](sampling/io/make_fakes.py)

#### Real-World Examples
Machine Learning examples on open-source datasets (local and AI Platform):
* [Acquire Valued Shoppers](examples/acquire-valued-shoppers)
* [Million Songs](examples/million-songs)

## Getting Started

### Installation
* Clone this repository

```bash
git clone https://github.com/teamdatatonic/tf-sampling.git
```

* Installing the `sampling` module from source:
```bash
python setup.py install
```

## Usage

### Run Tests
Execute tests from `sampling/tests/`:
```bash
python -m pytest
```

Generate coverage report:
```bash
python -m pytest --cov=../io .
```

## Contributing
If you'd like to contribute, please **fork the repository** and make changes as you'd like. **Pull requests** are welcome.

## Licensing
Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.
