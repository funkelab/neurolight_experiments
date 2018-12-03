#!/bin/bash

#conda activate maislightenv

export CUDA_VISIBLE_DEVICES="3"

python experiments/02_setup/setup02/mknet.py
python experiments/02_setup/setup02/train.py 20
