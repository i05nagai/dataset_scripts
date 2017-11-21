#!/bin/bash

path_to_this_dir=$(cd $(dirname ${0});pwd)
python -m misc.keras.cli \
  train \
  --fine_tune \
  --model_name resnet50 \
  --data_dir "${path_to_this_dir}/misc/keras/image"
