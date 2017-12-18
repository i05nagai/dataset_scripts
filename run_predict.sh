#!/bin/bash

path_to_this_dir=$(cd $(dirname ${0});pwd)
python \
  -m misc.keras.cli \
  predict \
  --fine_tune \
  --data_dir "${path_to_this_dir}/misc/keras/image/validation/open" \
  --model_name resnet50
  # "${path_to_this_dir}/misc/keras/image/validation/private/image.hitosara.com_gg_image_0006077076_0006077076E2_320.jpg" "${path_to_this_dir}/misc/keras/image/validation/private/image.hitosara.com_gg_image_0006073966_0006073966E2_320.jpg"
