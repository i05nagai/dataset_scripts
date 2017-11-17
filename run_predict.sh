#!/bin/bash

python -m misc.keras.cli --model vgg16 --predict "image/test/s_0n5d.jpg"
python -m misc.keras.cli --model vgg16 --predict "image/test/s_003o.jpg"
