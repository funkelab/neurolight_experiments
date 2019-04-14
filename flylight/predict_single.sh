#!/bin/bash

data_folder=$1
dataset=$2
start_idx=$3
num_samples=$4
exp=$5
setup=$6
iteration=$7
clipmax=$8
output_folder=$9

sample_list=${data_folder}/${dataset}.txt
readarray samples < $sample_list
files_per_worker=(${samples[@]:start_idx:num_samples})

echo "Taking" ${#files_per_worker[@]} "files out of" ${#samples[@]}
echo "Using Cuda device:" $CUDA_VISIBLE_DEVICES

expname=${setup}_$exp

for each in "${files_per_worker[@]}"
do
    echo $each
    output_file=$output_folder/$iteration/"$each".zarr
    echo $output_file
    
    if [ -f ${output_file} ]
	then
        echo ${i} "already created"
		continue
    fi
    python 03_predict/$setup/predict.py $expname $iteration $clipmax $each

done

echo "Finish worker"
