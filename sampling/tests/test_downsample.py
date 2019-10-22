"""
Unit tests for InputFnDownsampleWithWeight.

Usage:
python -m pytest

Coverage:
python -m pytest --cov=../io .
"""

import pytest
import numpy as np

import tensorflow as tf

from sampling.io.schema import Schema
from sampling.io.downsample import InputFnDownsampleWithWeight
from sampling.io.input_fn import Transform

DATA_DIR = 'test-data/'


@pytest.fixture
def schema():
    """Returns Schema object for million songs dataset."""
    schema_dict = {
        'user_name': 'STRING',
        'track_id': 'STRING',
        'n_listens': 'INTEGER',
    }
    return Schema.from_dict(schema_dict,
                            label='n_listens',
                            features_to_forward=['user_name', 'track_id'])


@pytest.fixture
def train_input_fn(schema, **kwargs):
    def _train_input_fn(**kwargs):
        return InputFnDownsampleWithWeight(positive_dir=DATA_DIR +
                                           'sample-01.csv',
                                           negative_dir=DATA_DIR + '*.csv',
                                           is_train=True,
                                           schema=schema,
                                           **kwargs)

    return _train_input_fn


@pytest.fixture
def eval_input_fn(schema, **kwargs):
    def _eval_input_fn(**kwargs):
        return InputFnDownsampleWithWeight(test_dir=DATA_DIR + '*.csv',
                                           is_train=False,
                                           schema=schema,
                                           shuffle=False,
                                           **kwargs)

    return _eval_input_fn


def generate_array(tensor):
    """Generate array from TensorFlow session."""
    with tf.Session() as sess:
        arr = tf.Session().run(tensor)
    return arr


@pytest.mark.parametrize('batch_size', [32, 1024])
@pytest.mark.parametrize('num_epochs', [1, 10])
@pytest.mark.parametrize('positive_size', [None, 1000, 10000])
@pytest.mark.parametrize('multiplier', [1, 5])
@pytest.mark.parametrize('weight', [0.0001, 100.0])
def test_input_fn_args(schema, train_input_fn, batch_size, num_epochs,
                       positive_size, multiplier, weight):
    """Test for attribute assignment and default values for input function."""
    input_fn = train_input_fn(batch_size=batch_size,
                              num_epochs=num_epochs,
                              positive_size=positive_size,
                              multiplier=multiplier,
                              weight=weight)

    assert input_fn.positive_dir == DATA_DIR + 'sample-01.csv'
    assert input_fn.negative_dir == DATA_DIR + '*.csv'
    assert input_fn.test_dir is None
    assert input_fn.schema == schema
    assert input_fn.is_train is True
    assert input_fn.shuffle is True
    assert input_fn.batch_size == batch_size
    assert input_fn.num_epochs == num_epochs
    assert input_fn.header is True
    assert input_fn.positive_size == positive_size
    assert input_fn.multiplier == multiplier
    assert input_fn.weight == weight


@pytest.mark.parametrize('positive_size,multiplier,exp_positive_size',
                         [(None, 2, 1000), (1000, 3, 1000), (20000, 10, 20000),
                          (200000, 20, 200000)])
def test_row_count(train_input_fn, positive_size, multiplier,
                   exp_positive_size):
    """Test _compute_dataset_rows and positive_rows."""
    input_fn = train_input_fn(multiplier=multiplier,
                              positive_size=positive_size)
    input_fn()

    assert input_fn.positive_size == exp_positive_size
    assert input_fn.negative_size == (exp_positive_size * multiplier)


@pytest.mark.parametrize(
    'shuffle,is_negative,multiplier,exp_transforms,exp_buffer,exp_take',
    [(True, True, 10, ['shuffle', 'take', 'interleave'], [[50]], [[11]]),
     (True, True, 20, ['shuffle', 'take', 'interleave'], [[50]], [[21]]),
     (False, True, 1, ['take', 'interleave'], [], [[5]]),
     (False, False, 10, ['interleave'], [], [])])
def test_separate_transforms(train_input_fn, shuffle, is_negative, multiplier,
                             exp_transforms, exp_buffer, exp_take):
    """Test _separate_transforms."""
    input_fn = train_input_fn(shuffle=shuffle,
                              multiplier=multiplier,
                              positive_size=1000)
    input_fn._extract()
    transforms = input_fn._separate_transforms(is_negative=is_negative)

    assert [t.name for t in transforms] == exp_transforms
    assert [t.args for t in transforms if t.name == 'shuffle'] == exp_buffer
    assert [t.args for t in transforms if t.name == 'take'] == exp_take


@pytest.mark.parametrize('batch_size', [256, 512, 1024])
@pytest.mark.parametrize('num_epochs', [1, 5, 10])
@pytest.mark.parametrize(
    'shuffle,multiplier,positive_size,exp_transforms,exp_buffer',
    [(True, 1, 1000, ['shuffle', 'batch', 'map', 'repeat'], [[2000]]),
     (True, 2, 1000, ['shuffle', 'batch', 'map', 'repeat'], [[3000]]),
     (True, 1, 20000000, ['shuffle', 'batch', 'map', 'repeat'], [[10000000]]),
     (False, 1, 1000, ['batch', 'map', 'repeat'], [])])
def test_combined_transforms(train_input_fn, batch_size, num_epochs, shuffle,
                             multiplier, positive_size, exp_transforms,
                             exp_buffer):
    """Test _combined_transforms."""
    input_fn = train_input_fn(shuffle=shuffle,
                              batch_size=batch_size,
                              num_epochs=num_epochs,
                              multiplier=multiplier,
                              positive_size=positive_size)
    input_fn()
    transforms = input_fn._combined_transforms()

    assert [t.name for t in transforms] == exp_transforms
    assert [t.args for t in transforms if t.name == 'shuffle'] == exp_buffer
    assert [t.args for t in transforms if t.name == 'batch'][0] == [batch_size]
    assert [t.args for t in transforms
            if t.name == 'repeat'][0] == [num_epochs]


@pytest.mark.parametrize('weight,exp_weight',
                         [(20.0, np.array([0.0476, 0.9524])),
                          (0.25, np.array([0.8, 0.2])),
                          (10000.0, np.array([0.0001, 0.9999])),
                          (0.0001, np.array([0.9999, 0.0001]))])
def test_normalize_weight(train_input_fn, weight, exp_weight):
    """Test weight is normalized correctly inside _normalize_weight."""
    input_fn = train_input_fn(weight=weight)
    weights = input_fn._normalize_weight()

    assert (weights == exp_weight).all()


@pytest.mark.parametrize('batch_size', [3, 5, 10])
@pytest.mark.parametrize('weight', np.random.uniform(0.0001, 10000.0, size=5))
def test_features(train_input_fn, batch_size, weight):
    input_fn = train_input_fn(shuffle=False,
                              batch_size=batch_size,
                              weight=weight,
                              positive_size=1000)
    features, labels = input_fn()
    weight_arr = generate_array(features['weight'])
    label_arr = generate_array(labels)

    weights = input_fn._normalize_weight()
    weight_exp = np.vectorize(lambda x: weights[0]
                              if x > 0 else weights[1])(label_arr)

    assert set(features.keys()) == set(
        ['user_name', 'track_id', 'user_name_', 'track_id_', 'weight'])
    assert (weight_arr == weight_exp).all()


@pytest.mark.parametrize('batch_size', [3, 5, 10])
@pytest.mark.parametrize('weight', np.random.uniform(0.0001, 10000.0, size=5))
def test_eval_features(eval_input_fn, batch_size, weight):
    input_fn = eval_input_fn(batch_size=batch_size, weight=weight)

    features, labels = input_fn()
    weight_arr = generate_array(features['weight'])

    assert set(features.keys()) == set(
        ['user_name', 'track_id', 'user_name_', 'track_id_', 'weight'])
    assert (weight_arr == np.ones(batch_size)).all()


@pytest.mark.parametrize('multiplier,exp_total', [(1, 2000), (2, 3000),
                                                  (5, 3000), (10, 3000),
                                                  (20, 3000)])
def test_concatenated_dataset(train_input_fn, multiplier, exp_total):
    """Test final size of dataset using _dataset_concat and _compute_dataset_rows."""
    input_fn = train_input_fn(multiplier=multiplier)

    positive_dataset, negative_dataset = input_fn._extract()
    positive_dataset = input_fn._transform_dataset(positive_dataset)
    negative_dataset = input_fn._transform_dataset(negative_dataset,
                                                   is_negative=True)
    dataset = input_fn._dataset_concat(positive_dataset, negative_dataset)
    total_size = input_fn._compute_dataset_rows(dataset)

    # maximum possible number of records is 3000 due to sample dataset size
    # total_size = positive_size + (positive_size * multiplier)
    assert total_size == exp_total
