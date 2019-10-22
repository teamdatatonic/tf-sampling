"""Input function for random majority undersampling and upweighting."""

import time
import functools
import numpy as np
import tensorflow as tf
from .input_fn import InputFn, Transform


def time_it(func):
    """Printing execution time of any function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('Time for executing {}: {} seconds'.format(
            func.__name__, round(end - start, 2)))
        return result

    return wrapper


class InputFnDownsampleWithWeight(InputFn):
    r"""Input function for random majority undersampling and upweighting.

    Downsamples majority class with specified multiplier, and applies weight.

    .. code-block:: python

       estimator.train(input_fn)

    Args:
        positive_dir (str):   Path to directory containing train data (only positive instances).
        negative_dir (str):   Path to directory containing train data (only negative instances).
        test_dir (str):       Path to directory containing test data for validation, evaluation, or prediction.
        schema (Schema):      As for `InputFn`.
        is_train (bool):      train_and_evaluate() loop or otherwise.
        shuffle (bool):       Enable shuffling of files / rows.
        batch_size (int):     Stacks n consecutive elements of dataset into single element.
        num_epochs (int):     Number of times to repeat.
        header (bool):        Skip first line for header if True.
        positive_size (int):  Total number of positive instances in dataset.
        multiplier (int):     Multiplier for downsampling (e.g. 1 is equal rows as positive instances).
        weight (float):       Weight to apply to every negative instance - set to 1.0 for equal weight.
    """
    def __init__(self,
                 positive_dir=None,
                 negative_dir=None,
                 test_dir=None,
                 schema=None,
                 is_train=False,
                 shuffle=True,
                 batch_size=128,
                 num_epochs=1,
                 header=True,
                 positive_size=None,
                 multiplier=1,
                 weight=1.0):

        self.positive_dir = positive_dir
        self.negative_dir = negative_dir
        self.test_dir = test_dir
        self.schema = schema
        self.is_train = is_train
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.header = header
        self.positive_size = positive_size
        self.multiplier = multiplier
        self.weight = weight

    def __call__(self):
        """Returns feeders of features and labels for passing to model_fn."""
        if self.is_train:
            positive_dataset, negative_dataset = self._extract()
            dataset = self._transform(positive_dataset, negative_dataset)

        else:
            test_dataset = self._extract()
            dataset = self._transform(test_dataset)

        features, labels = self._load(dataset)

        return features, labels

    @staticmethod
    def _create_dataset(filenames):
        """Makes dataset (of filenames) from filename glob patterns."""
        # Extract lines from input files using the Dataset API.
        file_list = tf.gfile.Glob(filenames)
        assert len(file_list) > 0, \
            'glob pattern {} did not match any files'.format(filenames)
        dataset = tf.data.Dataset.from_tensor_slices(file_list)
        return dataset, file_list

    def _extract(self):
        """Creates positive and negative datasets."""
        if self.is_train:
            positive_dataset, self.positive_list = self._create_dataset(
                self.positive_dir)
            negative_dataset, _ = self._create_dataset(self.negative_dir)
            return positive_dataset, negative_dataset
        test_dataset, _ = self._create_dataset(self.test_dir)
        return test_dataset

    @staticmethod
    def _filename_to_rows_fn(header=True):
        """Generate Dataset comprising lines from one or more text files."""
        def f(filepath):
            dataset = tf.data.TextLineDataset(filepath)
            if header:
                dataset = dataset.skip(1)
            return dataset

        return f

    @staticmethod
    @time_it
    def _compute_dataset_rows(dataset):
        """Computes number of rows in dataset."""
        reducer = tf.contrib.data.Reducer(init_func=lambda _: 0,
                                          reduce_func=lambda x, _: x + 1,
                                          finalize_func=lambda x: x)
        rows = tf.contrib.data.reduce_dataset(dataset, reducer)
        return int(tf.Session().run(rows))

    @property
    def positive_rows(self):
        """Property returning number of positive instances."""
        return self.positive_size

    @positive_rows.setter
    def positive_rows(self, positive_dataset):
        """Computes and sets number of positive instances."""
        self.positive_size = self._compute_dataset_rows(positive_dataset)

    def _separate_transforms(self, is_negative=False):
        """Transform operations depending on positive / negative / test datasets."""
        transforms = []
        if self.shuffle:
            transforms.append(Transform('shuffle', [50]))
        if is_negative:
            limit = max(5, (self.multiplier + 1) * len(self.positive_list))
            transforms.append(Transform('take', [limit]))
        transforms.append(
            Transform(
                'interleave', [self._filename_to_rows_fn(header=self.header)],
                dict(cycle_length=8,
                     block_length=8,
                     num_parallel_calls=tf.contrib.data.AUTOTUNE)))
        return transforms

    def _combined_transforms(self):
        """Transform operations for complete dataset."""
        transforms = []
        if self.shuffle:
            transforms.append(
                Transform(
                    'shuffle',
                    [min(self.negative_size + self.positive_size, 10000000)]))
        transforms.extend([
            Transform('batch', [self.batch_size]),
            Transform('map', [self.schema.parse_csv],
                      dict(num_parallel_calls=tf.contrib.data.AUTOTUNE)),
            Transform('repeat', [self.num_epochs])
        ])
        return transforms

    @staticmethod
    def _apply_transform(dataset, transforms):
        """Application of transforms on dataset."""
        for t in transforms:
            method = getattr(dataset, t.name)
            dataset = method(*t.args, **t.kwargs)
        return dataset

    def _transform_dataset(self, dataset, is_negative=False,
                           is_combined=False):
        """Generating transforms and applying to dataset."""
        if is_combined:
            transforms = self._combined_transforms()
        else:
            transforms = self._separate_transforms(is_negative)
        dataset = self._apply_transform(dataset, transforms)
        return dataset

    def _dataset_concat(self, positive_dataset, negative_dataset):
        """Concantenating positive and negative datasets after downsampling."""
        if not self.positive_size:
            self.positive_rows = positive_dataset
        self.negative_size = self.positive_size * self.multiplier
        negative_dataset = negative_dataset.take(self.negative_size)
        dataset = positive_dataset.concatenate(negative_dataset)
        return dataset

    def _transform(self, dataset1, dataset2=None):
        """Performs dataset transforms in sequence."""

        if self.is_train:
            positive_dataset = self._transform_dataset(dataset1)
            negative_dataset = self._transform_dataset(dataset2,
                                                       is_negative=True)
            dataset = self._dataset_concat(positive_dataset, negative_dataset)
        else:
            dataset = self._transform_dataset(dataset1)

        dataset = self._transform_dataset(dataset, is_combined=True)
        return dataset

    def _normalize_weight(self):
        """Normalizes weights for each class and rounds."""
        w = np.array([1.0, self.weight])
        return np.around(w / np.sum(w), 4)

    def _load(self, dataset):
        """Defines how to yield batches from dataset."""

        dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()

        if self.schema:
            for ff in self.schema.features_to_forward:
                features[ff + '_'] = tf.identity(features[ff])

        if self.is_train:
            w = self._normalize_weight()
            features['weight'] = (tf.where_v2(
                tf.equal(tf.cast(labels, tf.int64), 0), w[1], w[0]))
        else:
            feature_columns = [
                col for col in self.schema.names if col != self.schema.label
            ]
            features['weight'] = tf.ones_like(features[feature_columns[0]],
                                              dtype=tf.float32)

        return features, labels
