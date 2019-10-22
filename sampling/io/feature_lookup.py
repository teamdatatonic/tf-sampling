"""Feature lookup objects for data preparation in input functions.

See :class:`~datatonicml.io.sequence_input_fn.SequenceInputFn` for a working example.
"""

import tensorflow as tf
import collections


class FuncLookup:
    """User-defined (e.g. dummy) lookup for features or labels"""

    def __init__(self, lookup_func):
        self.lookup_func = lookup_func

    def lookup(self, key):
        return self.lookup_func(key)

    def init(self):
        pass


class FeatureLookup(collections.UserDict):
    """HashTable lookup for features by key (e.g. item id)."""

    def __init__(self, filename, schema, key_column=0, vocab_size=None, delimiter='\t'):
        """Reads features into HashTables for lookup at runtime.

        Args:
            filename:       A headerless csv-like file containing features per id (and no empty values)
            schema:         :class:`~datatonicml.io.schema.Schema` for this file.
            key_column:     Position of the id (key) column.
            vocab_size:     Number of rows in the file (if known).
            delimiter:      Field delimiter in file (NB the reader is not smart and cannot distinguish quoted delimiters).
        """

        super().__init__()

        self.filename = filename
        self.schema = schema
        self.key_column = key_column
        self.vocab_size = vocab_size
        self.delimiter = delimiter


    def init(self):
        for i, field in enumerate(self.schema):
            if i != self.key_column:
                self[field.name] = self._make_hashtable(field.name, i)


    def _make_hashtable(self, feature_name, column_index):
        return tf.contrib.lookup.HashTable(
            tf.contrib.lookup.TextFileInitializer(
                self.filename,
                self.schema[self.key_column].dtype,
                self.key_column,
                self.schema[column_index].dtype,
                column_index,
                vocab_size=self.vocab_size,
                delimiter=self.delimiter,
                name='hashtable_{}'.format(feature_name)
            ),
            default_value=tf.zeros([], dtype=self.schema[column_index].dtype)
        )


    def lookup(self, key):
        return {name:ht.lookup(key) for name, ht in self.items()}


class LabelVocabLookup:
    """Vocab map for categorical output."""

    def __init__(self, vocab_file, key_dtype=tf.string):
        self.vocab_file = vocab_file
        self.key_dtype = key_dtype

    def init(self):
        self.hashtable = self._make_hashtable()

    def _make_hashtable(self):
        return tf.contrib.lookup.HashTable(
            tf.contrib.lookup.TextFileInitializer(
                self.vocab_file,
                self.key_dtype, 0,
                tf.int64, tf.contrib.lookup.TextFileIndex.LINE_NUMBER,
                delimiter=','
            ),
            default_value=0
        )

    def lookup(self, key):
        return self.hashtable.lookup(key)
