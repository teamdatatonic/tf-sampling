from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import json
import argparse
import tensorflow as tf

from sampling.io.downsample import InputFnDownsampleWithWeight

from .util import (custom_weight_type, custom_int_type, create_schema,
                   export_parameters, export_results)

RANDOM_SEED = 42
tf.random.set_random_seed(RANDOM_SEED)


def get_args():
    """Argument parser returning dictionary of arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=['train', 'evaluate', 'predict'],
                        help='mode - train/evaluate/predict')
    parser.add_argument('--project', type=str, help='name of the GCP project')
    parser.add_argument('--bucket', type=str, help='name of the GCS bucket')
    parser.add_argument('--schema_path', type=str, help='path to schema json')
    parser.add_argument('--credentials_path',
                        type=str,
                        help='path to credentials json')
    parser.add_argument('--positive_dir',
                        type=str,
                        help='path to train data (positive instances)')
    parser.add_argument('--negative_dir',
                        type=str,
                        help='path to train data (negative instances)')
    parser.add_argument('--test_dir',
                        type=str,
                        help='path to test data for evaluation or prediction')
    parser.add_argument('--positive_size',
                        type=int,
                        help='number of positive instances')
    parser.add_argument('--multiplier',
                        type=custom_int_type,
                        default=1,
                        help='multiplier for downsampling')
    parser.add_argument('--weight',
                        type=custom_weight_type,
                        default=1.0,
                        help='weight to apply to every negative instance')
    parser.add_argument('--num_epochs',
                        type=custom_int_type,
                        default=1,
                        help='number of epochs to train for')
    parser.add_argument('--batch_size',
                        type=custom_int_type,
                        default=128,
                        help='batch size while training')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='learning rate while training')
    parser.add_argument('--optimizer',
                        type=str,
                        default='Adam',
                        choices=['Adam', 'Adagrad', 'ProximalAdagrad'],
                        help='GD optimizer')
    parser.add_argument('--hidden_units',
                        type=str,
                        default='64,32',
                        help='number of hidden units per hidden layer')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        help='dropout fraction')
    parser.add_argument('--cloud',
                        dest='cloud',
                        action='store_true',
                        help='execute on AI Platform')
    parser.add_argument('--job-dir',
                        type=str,
                        help='working directory for models and checkpoints')

    args, _ = parser.parse_known_args()
    return args


def metrics(labels, predictions):
    """Define evaluation metrics."""
    return {
        'accuracy': tf.metrics.accuracy(labels, predictions['class_ids']),
        'precision': tf.metrics.precision(labels, predictions['class_ids']),
        'recall': tf.metrics.recall(labels, predictions['class_ids']),
        'f1': tf.contrib.metrics.f1_score(labels, predictions['class_ids']),
        'auc': tf.metrics.auc(labels, predictions['logistic'])
    }


def build_feature_columns(schema):
    """Build feature columns as input to the model."""

    # non-numeric columns
    exclude = ['customer_id', 'brand', 'promo_sensitive', 'weight', 'label']

    # numeric feature columns
    numeric_column_names = [col for col in schema.names if col not in exclude]
    numeric_columns = [
        tf.feature_column.numeric_column(col) for col in numeric_column_names
    ]

    # identity column
    identity_column = tf.feature_column.categorical_column_with_identity(
        key='promo_sensitive', num_buckets=2)
    # DNNClassifier only accepts dense columns
    indicator_column = tf.feature_column.indicator_column(identity_column)

    # numeric weight column
    weight_column = tf.feature_column.numeric_column('weight')

    feature_columns = numeric_columns + [indicator_column]
    return feature_columns, weight_column


def initialise_optimizer(optimizer, lr):
    """Define optimizer."""
    opt_dict = {
        'Adam': tf.train.AdamOptimizer(lr),
        'Adagrad': tf.train.AdagradOptimizer(lr),
        'ProximalAdagrad': tf.train.ProximalAdagradOptimizer(lr),
    }
    if opt_dict.get(optimizer):
        return opt_dict[optimizer]
    raise Exception('Optimizer {} not recognised'.format(optimizer))


def main(FLAGS):
    tf.logging.set_verbosity(tf.logging.INFO)

    features_to_forward = ['customer_id', 'brand']
    schema = create_schema(flags=FLAGS,
                           label='label',
                           features_to_forward=features_to_forward)

    run_config = tf.estimator.RunConfig(tf_random_seed=RANDOM_SEED,
                                        save_checkpoints_steps=1000,
                                        keep_checkpoint_max=3,
                                        save_summary_steps=1000)

    feature_columns, weight_column = build_feature_columns(schema)
    optimizer = initialise_optimizer(FLAGS.optimizer, FLAGS.learning_rate)

    estimator = tf.estimator.DNNClassifier(
        n_classes=2,
        hidden_units=FLAGS.hidden_units.split(','),
        feature_columns=feature_columns,
        optimizer=optimizer,
        activation_fn=tf.nn.relu,
        dropout=FLAGS.dropout,
        batch_norm=True,
        model_dir=FLAGS.job_dir + '/model',
        config=run_config,
        weight_column=weight_column)

    estimator = tf.contrib.estimator.add_metrics(estimator, metrics)
    estimator = tf.contrib.estimator.forward_features(
        estimator, keys=schema.features_to_forward)

    train_input_fn = InputFnDownsampleWithWeight(
        positive_dir=FLAGS.positive_dir,
        negative_dir=FLAGS.negative_dir,
        schema=schema,
        is_train=True,
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.num_epochs,
        positive_size=FLAGS.positive_size,
        multiplier=FLAGS.multiplier,
        weight=FLAGS.weight)

    test_input_fn = InputFnDownsampleWithWeight(test_dir=FLAGS.test_dir,
                                                schema=schema,
                                                shuffle=False,
                                                batch_size=2**15,
                                                num_epochs=1)

    if FLAGS.mode == 'train':
        tf.estimator.train_and_evaluate(
            estimator,
            tf.estimator.TrainSpec(train_input_fn),
            tf.estimator.EvalSpec(
                test_input_fn,
                steps=100,  # 'None' to evaluate on entire dataset
                start_delay_secs=10,
                throttle_secs=10))

        estimator.export_savedmodel(
            export_dir_base=FLAGS.job_dir + '/serving',
            serving_input_receiver_fn=schema.serving_input_receiver_fn)

        train_attr = {'positive_size': train_input_fn.positive_size}
        export_parameters(FLAGS, **train_attr)

    elif FLAGS.mode == 'evaluate':
        results = estimator.evaluate(
            test_input_fn,
            checkpoint_path=tf.train.latest_checkpoint(FLAGS.job_dir))
        export_results(FLAGS, results)

    # use for local predictions on a test set - for batch scoring use AI Platform predict
    elif FLAGS.mode == 'predict':
        predictions = estimator.predict(
            test_input_fn,
            checkpoint_path=tf.train.latest_checkpoint(FLAGS.job_dir))

        timestamp = datetime.datetime.utcnow().strftime('%Y%m%d-%H%M')
        file_name = 'predictions-{}.json'.format(timestamp)
        output_path = os.path.join(FLAGS.job_dir, file_name)

        with open(output_path, 'w') as json_output:
            for p in predictions:
                results = {
                    'customer_id': p['customer_id'].decode('utf-8'),
                    'brand': p['brand'].decode('utf-8'),
                    'predicted_label': int(p['class_ids'][0]),
                    'logistic': float(p['logistic'][0])
                }
                json_output.write(
                    json.dumps(results, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main(FLAGS=get_args())
