#!/usr/bin/env bash

# local training using AI Platform
# usage: bash local-ai-platform-train.sh <GCP PROJECT ID>

PROJECT=$1
BUCKET=tf-sampling
GCS_PATH=${BUCKET}/acquire-valued-shoppers
JOB_NAME=DNN_training_$(date +"%Y%m%d_%H%M")

gcloud ai-platform local train \
  --job-dir=gs://${GCS_PATH}/models/${JOB_NAME} \
  --package-path=ai-platform/trainer \
  --module-name=trainer.task \
  -- \
  --project=${PROJECT} \
  --bucket=${BUCKET} \
  --cloud \
  --mode=train \
  --schema_path=gs://${GCS_PATH}/schema.json \
  --positive_dir=gs://${GCS_PATH}/data/train/positive/*.csv \
  --negative_dir=gs://${GCS_PATH}/data/train/negative/*.csv \
  --test_dir=gs://${GCS_PATH}/data/dev/*.csv \
  --multiplier=1 \
  --weight=28.0 \
  --num_epochs=1 \
  --batch_size=256 \
  --learning_rate=0.001 \
  --optimizer=Adam \
  --hidden_units='64,32' \
  --dropout=0.2
