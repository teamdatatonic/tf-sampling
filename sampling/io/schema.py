"""Defines a Schema class for ingesting structured data."""

from google.cloud import bigquery, storage
import tensorflow as tf
import numpy as np
import json
import collections

from ..gcp.api import get_credentials
from ..util.dimensions import ensure_iterable

DTYPE_DICT = {
    'STRING': tf.string,
    'INTEGER': tf.int64,
    'FLOAT': tf.float32,
    'NUMERIC': tf.float32,
    'BOOLEAN': tf.bool,
    'TIMESTAMP': None,
    'RECORD': None
}

DEFAULT_DICT = {
    'STRING': ' ',
    'INTEGER': 0,
    'FLOAT': 0.0,
    'NUMERIC': 0.0,
    'BOOLEAN': False,
    'TIMESTAMP': None,
    'RECORD': None
}


class Field(bigquery.SchemaField):
    """Extends BigQuery SchemaField by binding defaults and tensorflow types.

    Note: does not currently handle TIMESTAMP, nested (RECORD) or REPEATED entries.
    """
    def __init__(self,
                 name,
                 field_type,
                 mode='NULLABLE',
                 description=None,
                 fields=()):

        self.dtype = _make_tf_dtype(field_type)
        bq_dtype = _make_bq_dtype(self.dtype)
        self.default = DEFAULT_DICT[bq_dtype]

        assert mode.upper() in ('NULLABLE','REQUIRED'), \
            'unsupported field mode {}'.format(mode.upper())

        super().__init__(name, bq_dtype, mode, description, fields)

    @classmethod
    def from_schemafield(cls, f):
        return cls(f.name, f.field_type, f.mode, f.description, f.fields)


class Schema(collections.UserList):
    """Schema (i.e. list of Fields) for reading tabular data.

    Can be built manually with list operations, or read from json.

    Args:
        label:      Name of field to use as label
        features_to_forward:  Name(s) of field(s) to copy for forwarding (so
                              that they can also be used in a model). Copied
                              features will be named with a trailing underscore,
                              so if you need to forward a field called 'user_id'
                              then the key for tf.contrib.estimator.forward_features
                              should be 'user_id_'.
    """
    def __init__(self, arg=None, label=None, features_to_forward=[]):
        if arg is None:
            super().__init__()
        else:
            super().__init__(arg)

        self.label = label
        self.features_to_forward = ensure_iterable(features_to_forward)

    def __getitem__(self, index):
        if isinstance(index, int):
            return super().__getitem__(index)
        else:
            return self.as_dict[index]

    @property
    def names(self):
        return [field.name for field in self]

    @property
    def defaults(self):
        return [[tf.constant(field.default, dtype=field.dtype)]
                for field in self]

    @property
    def dtypes(self):
        return [field.dtype for field in self]

    @property
    def placeholders(self):
        return {
            field.name: tf.placeholder(shape=[None],
                                       dtype=field.dtype,
                                       name='placeholder_' + field.name)
            for field in self
        }

    @property
    def as_dict(self):
        return {field.name: field for field in self}

    @classmethod
    def from_BQ(cls,
                dataset,
                table,
                label=None,
                credentials_json=None,
                features_to_forward=None):
        """Gets schema by interrogating table directly."""

        credentials = get_credentials(credentials_json)

        client = bigquery.Client(credentials.project_id,
                                 credentials=credentials)
        dataset_ref = client.dataset(dataset)
        table_ref = dataset_ref.table(table)
        table = client.get_table(table_ref)

        return cls([Field.from_schemafield(f) for f in table.schema],
                   label=label,
                   features_to_forward=None)

    @classmethod
    def from_gcs(cls,
                 path,
                 label=None,
                 credentials_json=None,
                 features_to_forward=None):
        """Reads json schema from file in Google Cloud Storage."""

        credentials = get_credentials(credentials_json)

        client = storage.Client(credentials.project_id,
                                credentials=credentials)

        bucket_name = path.split('/')[2]
        bucket = client.get_bucket(bucket_name)
        blob = bucket.get_blob(path.split(bucket_name + '/')[1])
        json_string = blob.download_as_string()

        return cls.from_json(json_string.decode('utf-8'),
                             label=label,
                             features_to_forward=features_to_forward)

    @classmethod
    def from_json(cls, json_string, label=None, features_to_forward=None):
        """Reads schema from json string.

        String can be e.g. saved output of::

            bq show --schema --format=prettyjson [PROJECT_ID]:[DATASET].[TABLE]

        and is expected to be structured something like

        .. code-block:: javascript

            [
             {'mode': 'NULLABLE',
              'name': 'customer_id',
              'type': 'INTEGER',
              'description': '...description...'
              },
             ...
            ]
        """

        return cls([
            Field.from_api_repr(f_dict) for f_dict in json.loads(json_string)
        ],
                   label=label,
                   features_to_forward=features_to_forward)

    @classmethod
    def from_dict(cls, schemadict, label=None, features_to_forward=None):
        """Makes schema from dictionary (``{name:type, ... }``) """

        schemalist = []
        for name, field_type in schemadict.items():
            f = Field(name, field_type)
            schemalist.append(f)

        return cls(schemalist,
                   label=label,
                   features_to_forward=features_to_forward)

    def to_json(self):
        """Writes schema to json string.

        See also :meth:`~datatonicml.io.schema.Schema.from_json`)
        """

        output = [field.to_api_repr() for field in self]
        return json.dumps(output)

    def serving_input_receiver_fn(self,
                                  receive_csv=False,
                                  csv_includes_label=True):
        """Makes a raw serving input receiver for ML Engine prediction.

        Returns a class that receives and passes forward a feature Tensor dict.

        Args:
            receive_csv:         Served model expects csv rows (can batch-input .csv file)
            csv_includes_label:  Served model expects (unused) label column in .csv
        """
        class FakeLenDict(dict):
            """Make input dictionary appear length>1 (hack the export signature)"""
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def __len__(self):
                return 2

        if receive_csv:
            receiver_tensors = FakeLenDict(
                input=tf.placeholder(shape=[None], dtype=tf.string))
            feature_tensors, _ = self.parse_csv(
                receiver_tensors["input"], include_label=csv_includes_label)
        else:
            receiver_tensors = self.placeholders
            feature_tensors = dict([
                (name, (_bool2str(t) if t.dtype == tf.bool else t))
                for name, t in receiver_tensors.items()
            ])

        for ff in self.features_to_forward:
            feature_tensors[ff + '_'] = tf.identity(feature_tensors[ff])

        return tf.estimator.export.ServingInputReceiver(
            feature_tensors, receiver_tensors)

    @property
    def _csv_defaults(self):
        return [(['true'] if d else ['false']) if isinstance(d[0], bool) else d
                for d in self.defaults]

    def parse_csv(self, records, include_label=True, **kwargs):
        """Makes features and labels from tensor of csv rowstrings."""

        if include_label:
            tensors = tf.decode_csv(records, self._csv_defaults, **kwargs)
        else:
            defaults_without_label = [
                y for (x, y) in zip(self.names, self._csv_defaults)
                if x != self.label
            ]

            tensors = tf.decode_csv(records, defaults_without_label, **kwargs)

        features = dict(zip(self.names, tensors))

        try:
            labels = features.pop(self.label)
            return features, labels
        except KeyError:
            return features, []


def _make_tf_dtype(field_type):
    """Handles various ways of passing type information.

    (tf.DType, np.dtype, type or string)
    """

    if isinstance(field_type, tf.DType):
        return field_type
    elif isinstance(field_type, type):
        return tf.as_dtype(np.dtype(field_type))
    elif isinstance(field_type, str):
        try:
            return tf.as_dtype(field_type)
        except TypeError:
            try:
                return tf.as_dtype(np.dtype(field_type))
            except TypeError:
                assert field_type.upper() in ('STRING','INTEGER','FLOAT','NUMERIC','BOOLEAN'), \
                    'unsupported field type {}'.format(field_type.upper())
                return DTYPE_DICT[field_type]
    else:
        raise TypeError(
            '{} is not a recognized datatype or type name'.format(field_type))


def _make_bq_dtype(tftype):
    """Makes schema type string from tf.DType"""

    if tftype is tf.string:
        return 'STRING'
    elif tftype.is_integer:
        return 'INTEGER'
    elif tftype.is_floating:
        return 'FLOAT'
    elif tftype.is_bool:
        return 'BOOLEAN'
    else:
        raise TypeError('unsupported dtype {}'.format(tftype))


def _bool2str(tensor):
    return tf.where(tensor, tf.fill(tf.shape(tensor), 'true'),
                    tf.fill(tf.shape(tensor), 'false'))
