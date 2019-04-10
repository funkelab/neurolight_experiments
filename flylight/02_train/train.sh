#!/bin/bash

exp=$1
setup=$2
iteration=$3
clipmax=$4
root=/groups/kainmueller/home/maisl/workspace/neurolight/experiments

echo $exp $setup $iteration $clipmax

expname=${setup}_$exp
output_folder=$root/$expname
mkdir $output_folder

train_folder=$output_folder/train
mkdir $train_folder

this_file="$(cd "$(dirname "$0")"; pwd)"/`basename "$0"`
setup_folder="$(cd "$(dirname "$0")"; pwd)"/$setup

cp $this_file $output_folder
cp $setup_folder/* $output_folder

python $setup/mknet.py $expname > $output_folder/mknet.log 2>&1

python $setup/train.py $iteration $expname $clipmax > $output_folder/train.log 2>&1

