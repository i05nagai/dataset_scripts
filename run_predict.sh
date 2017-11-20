#!/bin/bash

path_to_this_dir=$(cd $(dirname ${0});pwd)
python \
  -m misc.keras.cli \
  --model resnet50 \
  --predict "${path_to_this_dir}/misc/keras/image/test/s_0n5d.jpg" "${path_to_this_dir}/misc/keras/image/test/s_003o.jpg"

