#!/usr/bin/env bash

# model predict / evaluate - set --mode=evaluate or --mode=predict
# --job-dir should point to model checkpoints

# usage: bash local-predict-evaluate.sh

python -m ai-platform.trainer.task \
  --mode=evaluate \
  --schema_path=schema.json \
  --test_dir=data/test/*.csv \
  --learning_rate=0.001 \
  --optimizer=Adam \
  --hidden_units='64,32' \
  --dropout=0.2 \
  --job-dir=model/model
