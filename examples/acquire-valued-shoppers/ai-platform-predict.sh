#!/usr/bin/env bash

# use deployed serving model on AI Platform to run batch predictions
# usage: bash ai-platform-predict.sh

MODEL_NAME=acquire_valued_shoppers_dnn
VERSION_NAME=v1

BUCKET=tf-sampling

TIMESTAMP=$(date +"%Y%m%d_%H%M")
JOB_NAME=${MODEL_NAME}_predictions_${TIMESTAMP}
MAX_WORKER_COUNT=1

gcloud ai-platform jobs submit prediction ${JOB_NAME} \
	--data-format=text \
	--input-paths=gs://${BUCKET}/acquire-valued-shoppers/test-predictions/test.json \
	--output-path=gs://${BUCKET}/acquire-valued-shoppers/test-predictions/${MODEL_NAME}_${VERSION_NAME}/${JOB_NAME} \
	--region=europe-west1 \
	--model=${MODEL_NAME} \
	--version=${VERSION_NAME} \
	--max-worker-count=${MAX_WORKER_COUNT}
