"""Define an InputFn class to simplify the passing of data to
estimator.train/evaluate/predict methods through (features, labels) tuples.

When called by these methods, an InputFn instance will first source a dataset
using the file names given, then transform it, then finally iterate through it
on demand.
"""

import tensorflow as tf
from collections import namedtuple
import namedtupled as NT
from attrdict import AttrDict


class Transform(namedtuple('Transform', ['name','args','kwargs'])):
    """Contains dataset-transforming method name and its arguments.

    .. automethod:: __new__
    """

    def __new__(cls, name, args=[], kwargs={}):
        return super().__new__(cls, name, args, kwargs)

    @staticmethod
    def make_defaults(schema, shuffle=True, batch_size=128, num_epochs=1, paired=False):
        if schema is None:
            raise ValueError('Default dataset transforms require a Schema')

        transforms = []
        if shuffle:
            transforms.extend([
                Transform('shuffle', [50]),
                Transform('interleave', [_filename_to_rows_fn(header=True)],
                                    dict(cycle_length=8, block_length=8, num_parallel_calls=tf.contrib.data.AUTOTUNE)),
                Transform('shuffle', [100000])
            ])
        else:
            transforms.append(Transform('flat_map',[_filename_to_rows_fn(header=True)]))

        if batch_size is not None:
            transforms.append(Transform('batch', [batch_size], dict(drop_remainder=paired)))

        transforms.extend([
            Transform('map', [schema.parse_csv], dict(num_parallel_calls=tf.contrib.data.AUTOTUNE)),
            Transform('cache'),
            Transform('repeat', [num_epochs])
        ])

        return transforms


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        assert callable(self.iterator_initializer_func)
        self.iterator_initializer_func(session)


class InputFn:
    """An ``input_fn`` for passing to ``estimator.train()``/``.evaluate()``/``.predict()``.

    When called, this function defines how to provide features and labels
    (which are each a Tensor or dict of Tensors) to the ``model_fn``.

    For more complex input pipelines users are advised to make
    a subclass and reimplement ``_extract``, ``_transform`` and ``_load``.

    Input function is called *once*, but adds ops to the graph that return the
    next ``(features, labels)`` until the dataset is finished. Hence it will
    define an upper bound on training duration unless ``transforms`` contains
    ``Transform('repeat')`` (with no ``count`` argument).

    N.B. when serving a model on ML Engine, an ``input_fn`` is not used (instead,
    ``serving_input_receiver_fn`` feeds data in - see ``Schema`` class)

    Args:
        filenames: string or iterable of strings. Glob patterns matching
                   data file names (e.g. ``'gs://my-bucket/data/*.csv'`` or
                   ``['../london/bus_*.csv', '../madrid/bus_*.csv']`` )

    Keyword Args:
        schema:      a Schema instance, for parsing the data files
                     with default Transforms

        transforms:  if no schema is passed, user must pass a list of Transform
                     elements, describing dataset transformation operations
                     in order.

        paired:      boolean indicating if this input_fn should return paired batches
                     with a label indicating which of the pair is superior.
    """

    def __init__(self, filenames,
                 transforms='default',
                 schema=None,
                 batch_size=128,
                 shuffle=True,
                 num_epochs=1,
                 initializable=False,
                 paired=False):

        self.filenames = filenames
        self.initializable = initializable
        if self.initializable: self.init_hook = IteratorInitializerHook()
        self.paired = paired
        self.schema = schema

        if transforms == 'default':
            self.transforms = Transform.make_defaults(schema=schema,
                                                      shuffle=shuffle,
                                                      batch_size=batch_size,
                                                      num_epochs=num_epochs,
                                                      paired=paired)
        else:
            self.transforms = transforms


    def __call__(self):
        """Returns feeders of features and labels for passing to model_fn."""

        dataset = self._extract()
        dataset = self._transform(dataset)
        features, labels = self._load(dataset)

        return features, labels


    def _extract(self):
        """Makes a dataset (of filenames) from filename glob patterns."""

        # Extract lines from input files using the Dataset API.
        file_list = tf.gfile.Glob(self.filenames)
        assert len(file_list) > 0, \
            'glob pattern {} did not match any files'.format(self.filenames)
        dataset = tf.data.Dataset.from_tensor_slices(file_list)

        return dataset


    def _transform(self, dataset):
        """Performs dataset transforms in sequence."""

        for t in self.transforms:
            method = getattr(dataset, t.name)
            dataset = method(*t.args, **t.kwargs)

        return dataset


    def _load(self, dataset):
        """Defines how to yield batches from dataset."""

        dataset = dataset.prefetch(1)

        if self.initializable:
            iterator = dataset.make_initializable_iterator()
        else:
            iterator = dataset.make_one_shot_iterator()

        if self.paired:
            iterator.__class__ = _PairedIterator
            features, labels = iterator.get_next_pair()
        else:
            features, labels = iterator.get_next()

        if self.initializable:
            init_fn = lambda sess: sess.run(iterator.initializer)
            self.init_hook.iterator_initializer_func = init_fn

        if self.schema is not None:
            for ff in self.schema.features_to_forward:
                features[ff+'_'] = tf.identity(features[ff])

        return features, labels


def _filename_to_rows_fn(header=True):
    def f(filepath):
        dataset = tf.data.TextLineDataset(filepath)
        if header: dataset = dataset.skip(1)
        return dataset
    return f


class _PairedIterator(tf.data.Iterator):
    """Extends iterator with paired iterating method."""

    def get_next_pair(self):
        """Processes batch into paired examples for Boosted Trees Ranker."""

        batch_features, batch_labels = self.get_next()

        # split the batch into two equal halves
        split_tensors = [tf.split(v,2) for v in batch_features.values()]
        a_feat_tensors = [a for a,b in split_tensors]
        b_feat_tensors = [b for a,b in split_tensors]
        renamed_a_features = zip([('a.'+key) for key in batch_features.keys()],
                                 a_feat_tensors)
        renamed_b_features = zip([('b.'+key) for key in batch_features.keys()],
                                 b_feat_tensors)

        # twice as many features, prefixed with 'a.' and 'b.'
        features = {}
        features.update(dict(renamed_a_features))
        features.update(dict(renamed_b_features))

        # labels for classification: is example 'a' ranked better than 'b'?
        a_labels, b_labels = tf.split(batch_labels,2)
        labels = tf.greater(a_labels,b_labels)

        return features, labels
