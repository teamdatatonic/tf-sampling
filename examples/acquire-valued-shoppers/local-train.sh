#!/usr/bin/env bash

# model training locally
# usage: bash local-train.sh

python -m ai-platform.trainer.task \
  --mode=train \
  --schema_path=schema.json \
  --positive_dir=data/train/positive/*.csv \
  --negative_dir=data/train/negative/*.csv \
  --test_dir=data/dev/*.csv \
  --multiplier=1 \
  --weight=28.0 \
  --num_epochs=1 \
  --batch_size=256 \
  --learning_rate=0.001 \
  --optimizer=Adam \
  --hidden_units='64,32' \
  --dropout=0.2 \
  --job-dir=model
