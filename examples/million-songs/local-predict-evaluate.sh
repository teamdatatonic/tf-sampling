#!/usr/bin/env bash

# model predict / evaluate - set --mode=evaluate or --mode=predict
# --job-dir should point to model checkpoints

# usage: bash local-predict-evaluate.sh

python -m ai-platform.trainer.task \
  --mode=evaluate \
  --user_vocab=vocab/users.csv \
  --song_vocab=vocab/songs.csv \
  --test_dir=data/test/*.csv \
  --learning_rate=0.001 \
  --optimizer=Ftrl \
  --job-dir=model/model
