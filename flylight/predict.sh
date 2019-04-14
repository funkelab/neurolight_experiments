#!/bin/bash

exp=$1
setup=$2
iteration=$3
clipmax=$4
num_workers=$5
dataset=$6

root=/groups/kainmueller/home/maisl/workspace/neurolight/experiments
data_folder=/groups/kainmueller/home/maisl/workspace/patch_instance_segmentation/patch_instance_segmentation/01_data

echo $exp $setup $iteration $clipmax $num_workers $dataset

# create output folder and copy scripts
exp_name=${setup}_$exp
output_folder=$root/$exp_name/test
mkdir $output_folder

this_file="$(cd "$(dirname "$0")"; pwd)"/`basename "$0"`
setup_folder="$(cd "$(dirname "$0")"; pwd)"/03_predict/$setup

cp $this_file $output_folder
cp $setup_folder/* $output_folder

./predict_all.sh $exp $setup $iteration $clipmax $num_workers $output_folder $data_folder $dataset
python 04_evaluate/evaluate.py $exp_name $iteration $dataset
