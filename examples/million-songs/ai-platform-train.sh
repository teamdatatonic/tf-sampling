#!/usr/bin/env bash

# model training on AI Platform
# usage: bash ai-platform-train.sh <GCP PROJECT ID>

PROJECT=$1
BUCKET=tf-sampling
GCS_PATH=${BUCKET}/million-songs
JOB_NAME=LR_training_$(date +"%Y%m%d_%H%M")

gcloud ai-platform jobs submit training ${JOB_NAME}  \
  --scale-tier=BASIC \
  --job-dir=gs://${GCS_PATH}/models/${JOB_NAME} \
  --package-path=ai-platform/trainer \
  --module-name=trainer.task \
  --packages=../../dist/sampling-0.1-py3-none-any.whl \
  --region=europe-west1 \
  --runtime-version=1.14 \
  --python-version=3.5 \
  -- \
  --project=${PROJECT} \
  --bucket=${BUCKET} \
  --cloud \
  --mode=train \
  --user_vocab=gs://${GCS_PATH}/vocab/users.csv \
  --song_vocab=gs://${GCS_PATH}/vocab/songs.csv \
  --positive_dir=gs://${GCS_PATH}/data/train/positive/*.csv \
  --test_dir=gs://${GCS_PATH}/data/dev/*.csv \
  --positive_size=2160509 \
  --balance_ratio=0.5 \
  --num_epochs=1 \
  --batch_size=1024 \
  --learning_rate=0.001 \
  --optimizer=Ftrl
