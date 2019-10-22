#!/usr/bin/env bash
# usage: source setup.sh

GREEN=$(tput setaf 2)
NORMAL=$(tput sgr0)

printf "${GREEN}Creating and activating Python 3 virtual environment: tfs${NORMAL}\n"
virtualenv -p python3 tfs
source tfs/bin/activate

printf "${GREEN}Installing package dependencies.${NORMAL}\n"
cd ..
python setup.py install
python setup.py bdist_wheel

printf "${GREEN}Downloading data from tf-sampling bucket on Google Cloud Storage.${NORMAL}\n"
cd examples
gsutil -m cp -r gs://tf-sampling/acquire-valued-shoppers/data acquire-valued-shoppers
gsutil -m cp -r gs://tf-sampling/million-songs/data million-songs