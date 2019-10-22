from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import argparse
import json
import tensorflow as tf

from sampling.io.make_fakes import InputFnWithFakePairs
from sampling.io.input_fn import InputFn
from sampling.io.schema import Schema, Field

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
    parser.add_argument('--credentials_path',
                        type=str,
                        help='path to credentials json')
    parser.add_argument('--user_vocab', type=str, help='path to user vocab')
    parser.add_argument('--song_vocab', type=str, help='path to song vocab')
    parser.add_argument('--positive_dir',
                        type=str,
                        help='path to train data (positive instances)')
    parser.add_argument('--test_dir',
                        type=str,
                        help='path to test data for evaluation or prediction')
    parser.add_argument('--positive_size',
                        type=int,
                        help='number of positive instances')
    parser.add_argument('--balance_ratio',
                        type=float,
                        default=0.5,
                        help='proportion that should be positive examples')
    parser.add_argument('--rejection_resample',
                        dest='rejection_resample',
                        action='store_true',
                        help='rejection resample (t/f)')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=1,
                        help='number of epochs to train for')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='batch size while training')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='learning rate while training')
    parser.add_argument('--optimizer',
                        type=str,
                        default='Ftrl',
                        choices=['Ftrl', 'Adam', 'Adagrad'],
                        help='GD optimizer')
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
        'rmse':
        tf.metrics.root_mean_squared_error(labels, predictions['predictions']),
        'mae':
        tf.metrics.mean_absolute_error(labels, predictions['predictions']),
    }


def build_feature_columns(flags):
    """Build feature columns as input to the model."""
    users = tf.feature_column.categorical_column_with_vocabulary_file(
        'user_name', flags.user_vocab)
    tracks = tf.feature_column.categorical_column_with_vocabulary_file(
        'track_id', flags.song_vocab)
    return [users, tracks]


def initialise_optimizer(optimizer, lr):
    """Define optimizer."""
    opt_dict = {
        'Ftrl': tf.train.FtrlOptimizer(lr),
        'Adam': tf.train.AdamOptimizer(lr),
        'Adagrad': tf.train.AdagradOptimizer(lr),
    }
    if opt_dict.get(optimizer):
        return opt_dict[optimizer]
    raise Exception('Optimizer {} not recognised'.format(optimizer))


def main(FLAGS):
    tf.logging.set_verbosity(tf.logging.INFO)

    features_to_forward = ['user_name', 'track_id']
    fields = [
        Field('user_name', tf.string),
        Field('track_id', tf.string),
        Field('n_listens', tf.float32)
    ]
    schema = Schema(fields,
                    label='n_listens',
                    features_to_forward=features_to_forward)

    run_config = tf.estimator.RunConfig(tf_random_seed=RANDOM_SEED,
                                        save_checkpoints_steps=1000,
                                        keep_checkpoint_max=3,
                                        save_summary_steps=1000)

    feature_columns = build_feature_columns(FLAGS)
    optimizer = initialise_optimizer(FLAGS.optimizer, FLAGS.learning_rate)

    estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns,
                                             optimizer=optimizer,
                                             model_dir=FLAGS.job_dir +
                                             '/model',
                                             config=run_config)

    estimator = tf.contrib.estimator.add_metrics(estimator, metrics)
    estimator = tf.contrib.estimator.forward_features(
        estimator, keys=schema.features_to_forward)

    train_input_fn = InputFnWithFakePairs(
        filenames=FLAGS.positive_dir,
        positive_dataset_size=FLAGS.positive_size,
        user_key='user_name',
        item_key='track_id',
        schema=schema,
        balance_ratio=FLAGS.balance_ratio,
        rejection_resample=FLAGS.rejection_resample,
        num_epochs=FLAGS.num_epochs,
        batch_size=FLAGS.batch_size)

    test_input_fn = InputFn(filenames=FLAGS.test_dir,
                            schema=schema,
                            batch_size=2**15,
                            shuffle=False,
                            num_epochs=1)

    if FLAGS.mode == 'train':
        tf.estimator.train_and_evaluate(
            estimator,
            tf.estimator.TrainSpec(train_input_fn,
                                   hooks=[train_input_fn.init_hook]),
            tf.estimator.EvalSpec(
                test_input_fn,
                steps=100,  # 'None' to evaluate on entire dataset
                start_delay_secs=10,
                throttle_secs=10))

        estimator.export_savedmodel(
            export_dir_base=FLAGS.job_dir + '/serving',
            serving_input_receiver_fn=schema.serving_input_receiver_fn)

    elif FLAGS.mode == 'evaluate':
        results = estimator.evaluate(
            test_input_fn,
            checkpoint_path=tf.train.latest_checkpoint(FLAGS.job_dir))

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
                    'user_name': p['user_name'].decode('utf-8'),
                    'track_id': p['track_id'].decode('utf-8'),
                    'n_listens': int(p['predictions'][0]),
                }
                json_output.write(
                    json.dumps(results, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main(FLAGS=get_args())
