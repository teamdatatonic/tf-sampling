#!/usr/bin/env bash

# model training locally
# usage: bash local-train.sh

python -m ai-platform.trainer.task \
  --mode=train \
  --user_vocab=vocab/users.csv \
  --song_vocab=vocab/songs.csv \
  --positive_dir=data/train/positive/*.csv \
  --test_dir=data/dev/*.csv \
  --positive_size=2160509 \
  --balance_ratio=0.5 \
  --num_epochs=1 \
  --batch_size=1024 \
  --learning_rate=0.001 \
  --optimizer=Ftrl \
  --job-dir=model
