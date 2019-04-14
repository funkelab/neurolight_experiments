#!/bin/bash

if [ $# -eq 0 ]
then
    echo "WARNING: No arguments supplied!"
    echo "predict_all.sh <experiment> <iteration> <num_worker> <output_dir>"

    exp=030419_01
    setup=setup01
    iteration=100000
    clipmax=1000
    num_workers=2
    output_folder=/groups/kainmueller/home/maisl/workspace/neurolight/experiments/${setup}_$exp
    data_folder=/groups/kainmueller/home/maisl/workspace/patch_instance_segmentation/patch_instance_segmentation/01_data
    dataset=flylight_270319_val
    cuda_devices=(0 1)
else
    exp=$1
    setup=$2
    iteration=$3
    clipmax=$4
    num_workers=$5
    output_folder=$6
    data_folder=$7
    dataset=$8
    echo "exp:" $exp $setup $iteration $clipmax
    echo "num workers:" $num_workers
    echo "output folder: " $output_folder
    echo "dataset:" $data_folder $dataset
fi

# check cuda devices
if [ -z "$cuda_devices" ]
then
    if [ -z "$CUDA_VISIBLE_DEVICES" ]
    then
        echo "Please set either $ 5 <cuda_devices> or CUDA_VISIBLE_DEVICES!"
        exit 1
    else
        IFS=', ' read -r -a cuda_devices <<< "$CUDA_VISIBLE_DEVICES"
    fi
fi
echo "Using cuda devices:" ${cuda_devices[@]}

# get samples in validation set from txt file
sample_list=${data_folder}/${dataset}.txt
readarray samples < $sample_list
num_samples=${#samples[@]}
num_samples_per_worker=$((num_samples / num_workers))
echo "Total files:" $num_samples "/ files per worker:" $num_samples_per_worker

for i in $(seq 1 $num_workers);
do
    idx=$((i-1))
    start_idx=$((idx * num_samples_per_worker))
    worker_log_file=$output_folder/predict_worker_${i}.out

    if [ $i -eq $num_workers ]
    then
        num_samples_per_worker=$((num_samples_per_worker + (num_samples - num_workers * num_samples_per_worker)))
    fi

    echo "Starting worker" $i "with start id" $start_idx
    echo "log file:" $worker_log_file

    export CUDA_VISIBLE_DEVICES=${cuda_devices[$idx]}

    ./predict_single.sh $data_folder $dataset $start_idx $num_samples_per_worker $exp $setup $iteration $clipmax $output_folder > $worker_log_file 2>&1 &

done
wait