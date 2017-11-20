#!/bin/bash

path_to_this_dir=$(cd $(dirname ${0});pwd)
python \
  -m misc.keras.cli \
  predict \
  --fine_tune \
  "${path_to_this_dir}/misc/keras/image/test/s_0n5d.jpg" "${path_to_this_dir}/misc/keras/image/test/s_003o.jpg" \
  --model_name resnet50
