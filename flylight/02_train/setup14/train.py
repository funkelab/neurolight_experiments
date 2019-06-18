from __future__ import print_function
import gunpowder as gp
import neurolight.gunpowder as nl
import numpy as np
import json
import logging
import os
import tensorflow as tf
import h5py
import sys
import math
from datetime import datetime

data_dir = '/home/maisl/data/flylight'
data_files = [
    'flylight_v1_train.hdf',
]

class Convert(gp.BatchFilter):

    def __init__(self, array, dtype):

        self.array = array
        self.dtype = dtype

    def process(self, batch, request):

        if self.array not in batch.arrays:
            return

        array = batch.arrays[self.array]

        try:
            array.data = array.data.astype(self.dtype)
        except:
            print("Cannot convert!")

        array.spec.dtype = self.dtype


class BinarizeLabels(gp.BatchFilter):

    def __init__(self, labels, labels_binary):

        self.labels = labels
        self.labels_binary = labels_binary

    def setup(self):

        spec = self.spec[self.labels].copy()
        spec.dtype = np.uint8
        self.provides(self.labels_binary, spec)

    def process(self, batch, request):

        spec = batch[self.labels].spec.copy()
        spec.dtype = np.uint8

        reduce_channels = np.max(batch[self.labels].data > 0, axis=0).astype(np.uint8)

        binarized = gp.Array(
            data=reduce_channels,
            spec=spec)

        batch[self.labels_binary] = binarized


def train_until(max_iteration, name='train_net', output_folder='.', clip_max=2000):

    # get the latest checkpoint
    if tf.train.latest_checkpoint(output_folder):
        trained_until = int(tf.train.latest_checkpoint(output_folder).split('_')[-1])
    else:
        trained_until = 0
        if trained_until >= max_iteration:
            return

    with open(os.path.join(output_folder, name + '_config.json'), 'r') as f:
        net_config = json.load(f)
    with open(os.path.join(output_folder, name + '_names.json'), 'r') as f:
        net_names = json.load(f)

    # array keys
    raw = gp.ArrayKey('RAW')
    gt_instances = gp.ArrayKey('GT_INSTANCES')
    gt_mask = gp.ArrayKey('GT_MASK')
    pred_mask = gp.ArrayKey('PRED_MASK')
    #loss_weights = gp.ArrayKey('LOSS_WEIGHTS')
    loss_gradients = gp.ArrayKey('LOSS_GRADIENTS')

    # array keys for base and add volume
    raw_base = gp.ArrayKey('RAW_BASE')
    gt_instances_base = gp.ArrayKey('GT_INSTANCES_BASE')
    gt_mask_base = gp.ArrayKey('GT_MASK_BASE')
    raw_add = gp.ArrayKey('RAW_ADD')
    gt_instances_add = gp.ArrayKey('GT_INSTANCES_ADD')
    gt_mask_add = gp.ArrayKey('GT_MASK_ADD')

    voxel_size = gp.Coordinate((1, 1, 1))
    input_shape = gp.Coordinate(net_config['input_shape'])
    output_shape = gp.Coordinate(net_config['output_shape'])
    context = gp.Coordinate(input_shape - output_shape) / 2

    request = gp.BatchRequest()
    request.add(raw, input_shape)
    request.add(gt_instances, output_shape)
    request.add(gt_mask, output_shape)
    #request.add(loss_weights, output_shape)
    request.add(raw_base, input_shape)
    request.add(raw_add, input_shape)
    request.add(gt_mask_base, output_shape)
    request.add(gt_mask_add, output_shape)

    snapshot_request = gp.BatchRequest()
    snapshot_request.add(raw, input_shape)
    #snapshot_request.add(raw_base, input_shape)
    #snapshot_request.add(raw_add, input_shape)
    snapshot_request.add(gt_mask, output_shape)
    #snapshot_request.add(gt_mask_base, output_shape)
    #snapshot_request.add(gt_mask_add, output_shape)
    snapshot_request.add(pred_mask, output_shape)
    snapshot_request.add(loss_gradients, output_shape)

    # specify data source
    # data source for base volume
    data_sources_base = tuple()
    for data_file in data_files:
        current_path = os.path.join(data_dir, data_file)
        with h5py.File(current_path, 'r') as f:
            data_sources_base += tuple(
                gp.Hdf5Source(
                    current_path,
                    datasets={
                        raw_base: sample + '/raw',
                        gt_instances_base: sample + '/gt',
                        gt_mask_base: sample + '/fg',
                    },
                    array_specs={
                        raw_base: gp.ArraySpec(interpolatable=True, dtype=np.uint16, voxel_size=voxel_size),
                        gt_instances_base: gp.ArraySpec(interpolatable=False, dtype=np.uint16, voxel_size=voxel_size),
                        gt_mask_base: gp.ArraySpec(interpolatable=False, dtype=np.bool, voxel_size=voxel_size),
                    }
                ) +
                Convert(gt_mask_base, np.uint8) +
                gp.Pad(raw_base, context) +
                gp.Pad(gt_instances_base, context) +
                gp.Pad(gt_mask_base, context) +
                gp.RandomLocation(min_masked=0.005,  mask=gt_mask_base)
                #gp.Reject(gt_mask_base, min_masked=0.005, reject_probability=1.)
                for sample in f)
    data_sources_base += gp.RandomProvider()

    # data source for add volume
    data_sources_add = tuple()
    for data_file in data_files:
        current_path = os.path.join(data_dir, data_file)
        with h5py.File(current_path, 'r') as f:
            data_sources_add += tuple(
                gp.Hdf5Source(
                    current_path,
                    datasets={
                        raw_add: sample + '/raw',
                        gt_instances_add: sample + '/gt',
                        gt_mask_add: sample + '/fg',
                    },
                    array_specs={
                        raw_add: gp.ArraySpec(interpolatable=True, dtype=np.uint16, voxel_size=voxel_size),
                        gt_instances_add: gp.ArraySpec(interpolatable=False, dtype=np.uint16, voxel_size=voxel_size),
                        gt_mask_add: gp.ArraySpec(interpolatable=False, dtype=np.bool, voxel_size=voxel_size),
                    }
                ) +
                Convert(gt_mask_add, np.uint8) +
                gp.Pad(raw_add, context) +
                gp.Pad(gt_instances_add, context) +
                gp.Pad(gt_mask_add, context) +
                gp.RandomLocation() +
                gp.Reject(gt_mask_add, min_masked=0.005, reject_probability=0.95)
                for sample in f)
    data_sources_add += gp.RandomProvider()
    data_sources = tuple([data_sources_base, data_sources_add]) + gp.MergeProvider()

    pipeline = (
            data_sources +
            nl.FusionAugment(
                raw_base, raw_add, gt_instances_base, gt_instances_add, raw, gt_instances,
                blend_mode='labels_mask', blend_smoothness=5, num_blended_objects=0
            ) +
            BinarizeLabels(gt_instances, gt_mask) +
            nl.Clip(raw, 0, clip_max) +
            gp.Normalize(raw, factor=1.0/clip_max) +
            gp.ElasticAugment(
                control_point_spacing=[20, 20, 20],
                jitter_sigma=[1, 1, 1],
                rotation_interval=[0, math.pi/2.0],
                subsample=4) +
            gp.SimpleAugment(mirror_only=[1, 2], transpose_only=[1, 2]) +

            gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1) +
            gp.IntensityScaleShift(raw, 2, -1) +
            #gp.BalanceLabels(gt_mask, loss_weights) +

            # train
            gp.PreCache(
                cache_size=40,
                num_workers=10) +
            gp.tensorflow.Train(
                os.path.join(output_folder, name),
                optimizer=net_names['optimizer'],
                loss=net_names['loss'],
                inputs={
                    net_names['raw']: raw,
                    net_names['gt']: gt_mask,
                    #net_names['loss_weights']: loss_weights,
                },
                outputs={
                    net_names['pred']: pred_mask,
                },
                gradients={
                    net_names['output']: loss_gradients,
                },
                save_every=5000) +

            # visualize
            gp.Snapshot({
                    raw: 'volumes/raw',
                    pred_mask: 'volumes/pred_mask',
                    gt_mask: 'volumes/gt_mask',
                    #loss_weights: 'volumes/loss_weights',
                    loss_gradients: 'volumes/loss_gradients',
                },
                output_filename=os.path.join(output_folder, 'snapshots', 'batch_{iteration}.hdf'),
                additional_request=snapshot_request,
                every=2500) +
            gp.PrintProfilingStats(every=1000)
    )

    with gp.build(pipeline):
        
        print("Starting training...")
        for i in range(max_iteration - trained_until):
            pipeline.request_batch(request)


if __name__ == "__main__":

    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    iteration = 100
    root = '/home/maisl/workspace/neurolight/experiments'
    experiment = 'setup14_020519_00'
    name = 'train_net'
    clip_max = 1500

    if len(sys.argv) > 1:
        iteration = int(sys.argv[1])
        experiment = sys.argv[2]

    if len(sys.argv) > 3:
        clip_max = int(sys.argv[3])

    if len(sys.argv) > 4:
        root = sys.argv[4]

    output_folder = os.path.join(root, experiment, 'train')
    try:
        os.makedirs(os.path.join(output_folder, 'snapshots'))
    except OSError:
        pass

    logging.basicConfig(level=logging.INFO)
    start = datetime.now()
    
    print('start training with: ', experiment, iteration, name, clip_max, output_folder)
    train_until(iteration, name, output_folder, clip_max)

    print('time: ', datetime.now() - start)

