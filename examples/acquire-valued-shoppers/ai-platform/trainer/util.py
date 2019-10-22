import os
import datetime
import json
import argparse

from google.cloud import storage

from sampling.io.schema import Schema
from sampling.gcp.api import get_credentials


def custom_weight_type(value):
    """Weight column should be float value within a particular range."""
    value = float(value)
    min_ = 1e-4
    max_ = 1e4
    if value < min_ or value > max_:
        raise argparse.ArgumentTypeError('{} not in range [{}, {}]'.format(
            value, min_, max_))
    return value


def custom_int_type(value):
    """Integer columns with positive values."""
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError(
            '{} must be a positive integer'.format(value))
    return value


def create_schema(flags, label=None, features_to_forward=None):
    """Create Schema object from JSON."""
    if flags.cloud:
        return Schema.from_gcs(flags.schema_path,
                               label=label,
                               features_to_forward=features_to_forward,
                               credentials_json=flags.credentials_path)
    else:
        with open(flags.schema_path, 'r') as schema:
            json_string = schema.read()
        return Schema.from_json(json_string=json_string,
                                label=label,
                                features_to_forward=features_to_forward)


def upload_to_gcs(flags, filename, data):
    """Upload files to Cloud Storage."""
    credentials = get_credentials(flags.credentials_path)
    client = storage.Client(flags.project, credentials=credentials)
    bucket = client.get_bucket(flags.bucket)
    path = flags.job_dir.replace('gs://{}/'.format(flags.bucket), '')
    path = os.path.join(path, filename)
    json_data = json.dumps(data)
    blob = bucket.blob(path)
    blob.upload_from_string(json_data)


def export_parameters(flags, **kwargs):
    """Generate JSON with model settings and evaluation metrics."""

    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d-%H%M')
    filename = 'train-parameters-{}.json'.format(str(timestamp))

    parameter_dict = flags.__dict__
    if kwargs:
        parameter_dict.update(kwargs)

    if flags.cloud:
        upload_to_gcs(flags, filename, parameter_dict)
    else:
        with open(os.path.join(flags.job_dir, filename), 'w') as write_file:
            write_file.write(
                json.dumps(parameter_dict, sort_keys=True, indent=2))


def export_results(flags, results):
    """Generate JSON with evaluation metrics."""

    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d-%H%M')
    filename = 'evaluation-{}.json'.format(timestamp)

    results_dict = {
        'job_dir': str(flags.job_dir),
        'accuracy': str(results['accuracy']),
        'precision': str(results['precision']),
        'recall': str(results['recall']),
        'f1': str(results['f1']),
        'auc': str(results['auc'])
    }

    if flags.cloud:
        upload_to_gcs(flags, filename, results_dict)
    else:
        with open(os.path.join(flags.job_dir, filename), 'w') as write_file:
            write_file.write(json.dumps(results_dict, sort_keys=True,
                                        indent=2))
