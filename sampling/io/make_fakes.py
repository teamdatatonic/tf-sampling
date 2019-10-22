"""Input function for recommenders, with fake user-item pairs.

.. note::
   MUST use this input_fn's init_hook to initialize when training with ``rejection_resample=True``:

   .. code-block:: python

      estimator.train(input_fn, hooks=[input_fn.init_hook])

"""

import tensorflow as tf
from tensorflow.contrib.lookup import HashTable, KeyValueTensorInitializer
import numpy as np
from .input_fn import InputFn, Transform


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""
    def __init__(self):
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        assert callable(self.iterator_initializer_func)
        self.iterator_initializer_func(session)


class InputFnWithFakePairs(InputFn):
    r"""Input function for recommenders, with fake user-item pairs.

    Generates user-item pairs by random combination followed by lookup to see if
    combination exists in true dataset. Does not need to read such pairs from disk.

    Currently only supports user-item ids as features.

    **MUST use init_hook to initialize when training with ``rejection_resample=True``**

    .. code-block:: python

       estimator.train(input_fn, hooks=[input_fn.init_hook])

    NB warnings will appear at the end of training due to selecting an exhausted
    dataset. Expect something like 10%.

    Solution for balanced sampling in 538's Riddler Express 2019-04-05 given at
    https://fivethirtyeight.com/features/how-many-times-a-day-is-a-broken-clock-right/

    .. math::

       E(k) \;=\; \sum_{k=0}^N k {2N-k \choose N} \left(0.5\right)^{2N-k}

    Args:
        filenames:      As for `InputFn`.
        user_key:       Key for the user_id column.
        item_key:       Key for the item_id column.
        user_feature_lookup:    FeatureLookup for user-keyed features
        item_feature_lookup:    FeatureLookup for item-keyed features
        default_label:  Default value to set label as for fake examples.
        balance_ratio:  Proportion of dataset that should be positive examples.
        transforms:     As for `InputFn`. Applied to positive dataset. Batching will be undone.
        schema:         As for `InputFn`.
        batch_size:     As for `InputFn`.
        shuffle:        As for `InputFn`.
        num_epochs:     As for `InputFn`.
        paired:         As for `InputFn`.
    """
    def __init__(self,
                 filenames,
                 positive_dataset_size,
                 user_key='user_id',
                 item_key='item_id',
                 user_feature_lookup=None,
                 item_feature_lookup=None,
                 default_label=0.0,
                 balance_ratio=0.5,
                 rejection_resample=False,
                 transforms='default',
                 schema=None,
                 batch_size=128,
                 shuffle=True,
                 num_epochs=1,
                 paired=False):

        self.positive_dataset_size = positive_dataset_size
        self.user_key = user_key
        self.item_key = item_key
        self.default_label = default_label
        self.balance_ratio = balance_ratio
        self.rejection_resample = rejection_resample
        self.user_feature_lookup = user_feature_lookup
        self.item_feature_lookup = item_feature_lookup

        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.init_hook = IteratorInitializerHook()

        if transforms == 'default':
            transforms = Transform.make_defaults(schema=schema,
                                                 shuffle=False,
                                                 batch_size=None,
                                                 num_epochs=1)

        super().__init__(filenames,
                         transforms=transforms,
                         schema=schema,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_epochs=num_epochs,
                         paired=paired)

    def _transform(self, dataset):
        """Performs dataset transforms in sequence."""

        positive_dataset = super()._transform(dataset)

        self._make_pair_presence_hashtable(positive_dataset)

        label_type = positive_dataset.output_types[-1]
        self.default_label = tf.cast(self.default_label, label_type)
        negative_dataset = self._cross_dataset

        if self.rejection_resample:
            negative_dataset = negative_dataset.apply(
                self._rejection_resampler)
            negative_dataset = negative_dataset.map(
                lambda x, y: y,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

        positive_dataset = positive_dataset.apply(
            tf.data.experimental.shuffle_and_repeat(100000,
                                                    count=self.num_epochs))

        balanced = tf.data.experimental.sample_from_datasets(
            [negative_dataset, positive_dataset],
            weights=[1 - self.balance_ratio, self.balance_ratio])

        def user_lookup(features, labels):
            features.update(
                self.user_feature_lookup.lookup(features[self.user_key]))
            return features, labels

        def item_lookup(features, labels):
            features.update(
                self.item_feature_lookup.lookup(features[self.item_key]))
            return features, labels

        if self.user_feature_lookup is not None:
            self.user_feature_lookup.init()
            balanced = balanced.map(
                user_lookup, num_parallel_calls=tf.contrib.data.AUTOTUNE)

        if self.item_feature_lookup is not None:
            self.item_feature_lookup.init()
            balanced = balanced.map(
                item_lookup, num_parallel_calls=tf.contrib.data.AUTOTUNE)

        final = balanced.batch(self.batch_size, drop_remainder=self.paired)

        return final

    def _load(self, dataset):
        """Defines how to yield batches from dataset."""

        dataset = dataset.prefetch(1)
        iterator = dataset.make_initializable_iterator()

        if self.paired:
            iterator.__class__ = _PairedIterator
            features, labels = iterator.get_next_pair()
        else:
            features, labels = iterator.get_next()

        init_fn = lambda sess: sess.run(iterator.initializer)
        self.init_hook.iterator_initializer_func = init_fn
        return features, labels

    def _make_pair_presence_hashtable(self, positive_dataset):
        iterator = positive_dataset.batch(self.positive_dataset_size +
                                          1).make_one_shot_iterator()
        positives = iterator.get_next()

        with tf.Session() as sess:
            features, labels = sess.run(positives)
            users = features[self.user_key]
            items = features[self.item_key]

            if self.rejection_resample:
                self._pair_presence_hashtable = HashTable(
                    KeyValueTensorInitializer(
                        users + b'@' + items,
                        tf.ones_like(users, dtype=tf.int32)),
                    default_value=0)
                self._pair_presence_hashtable.init.run(session=sess)

        self._positive_size = len(users)
        self._negative_size = int(self._positive_size *
                                  (1 - self.balance_ratio) /
                                  self.balance_ratio)
        self.users = tf.constant(np.unique(users))
        self.items = tf.constant(np.unique(items))
        cross_size = self.users.shape.as_list()[0] * self.items.shape.as_list(
        )[0]
        self._sparsity = self._positive_size / cross_size

    @property
    def _rejection_resampler(self):
        def class_func(x, labels):
            key = tf.string_join([x[self.user_key], x[self.item_key]],
                                 separator=b'@')
            return self._pair_presence_hashtable.lookup(key)

        rejection_resampler = tf.data.experimental.rejection_resample(
            class_func,
            target_dist=[1.0, 0.0],
            initial_dist=[(1 - self._sparsity), self._sparsity])

        return rejection_resampler

    @property
    def _cross_dataset(self):
        def random_pair(x):
            user_indices = tf.random.uniform([self._negative_size, 1],
                                             maxval=tf.size(self.users),
                                             dtype=tf.int32)

            item_indices = tf.random.uniform([self._negative_size, 1],
                                             maxval=tf.size(self.items),
                                             dtype=tf.int32)

            user_ids = tf.gather_nd(self.users, user_indices)
            item_ids = tf.gather_nd(self.items, item_indices)

            features = {self.user_key: user_ids, self.item_key: item_ids}

            return features, tf.fill([self._negative_size], self.default_label)

        dummy = tf.data.Dataset.from_tensors(0)
        random_pairs = dummy.map(random_pair, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                            .apply(tf.data.experimental.unbatch())
        cross_dataset = random_pairs.repeat(self.num_epochs)

        return cross_dataset
