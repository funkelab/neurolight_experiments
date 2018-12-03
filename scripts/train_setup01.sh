#!/bin/bash

#conda activate maislightenv

export CUDA_VISIBLE_DEVICES="4"

python experiments/02_setup/setup01/mknet.py
python experiments/02_setup/setup01/train.py 20
