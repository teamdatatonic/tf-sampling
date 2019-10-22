#!/usr/bin/env bash

# deploy model on AI Platform for serving - model location in Cloud Storage
# usage: bash ai-platform-deploy.sh <JOB NAME> <MODEL ID>

MODEL_NAME=acquire_valued_shoppers_dnn
VERSION_NAME=v1

BUCKET=tf-sampling

JOB_NAME=$1
MODEL_ID=$2

MODEL_PATH=gs://${BUCKET}/acquire-valued-shoppers/models/${JOB_NAME}/serving/${MODEL_ID}

gcloud ai-platform models create ${MODEL_NAME} \
    --regions=europe-west1

gcloud ai-platform versions create ${VERSION_NAME} \
    --model=${MODEL_NAME} \
    --origin=${MODEL_PATH} \
    --python-version=3.5 \
    --runtime-version=1.14 \
    --framework=tensorflow
