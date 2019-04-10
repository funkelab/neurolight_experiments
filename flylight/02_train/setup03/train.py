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


data_dir = '/groups/kainmueller/home/maisl/workspace/patch_instance_segmentation/patch_instance_segmentation/01_data'
data_files = [
    'flylight_270319_train.hdf',
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
    hsl = gp.ArrayKey('HSL')
    gt_mask = gp.ArrayKey('GT_MASK')
    pred_mask = gp.ArrayKey('PRED_MASK')
    loss_weights = gp.ArrayKey('LOSS_WEIGHTS')
    loss_gradient = gp.ArrayKey('LOSS_GRADIENT')

    voxel_size = gp.Coordinate((1, 1, 1))
    input_shape = gp.Coordinate(net_config['input_shape'])
    output_shape = gp.Coordinate(net_config['output_shape'])
    context = gp.Coordinate(input_shape - output_shape) / 2

    request = gp.BatchRequest()
    request.add(raw, input_shape)
    request.add(hsl, input_shape)
    request.add(gt_mask, output_shape)
    request.add(loss_weights, output_shape)

    snapshot_request = gp.BatchRequest()
    snapshot_request.add(raw, input_shape)
    snapshot_request.add(hsl, input_shape)
    snapshot_request.add(gt_mask, output_shape)
    snapshot_request.add(pred_mask, output_shape)
    snapshot_request.add(loss_gradient, output_shape)

    # specify data source
    data_sources = tuple()
    for data_file in data_files:
        current_path = os.path.join(data_dir, data_file)
        with h5py.File(current_path, 'r') as f:
            data_sources += tuple(
                gp.Hdf5Source(
                    current_path,
                    datasets={
                        raw: sample + '/raw',
                        gt_mask: sample + '/fg'
                    },
                    array_specs={
                        raw: gp.ArraySpec(interpolatable=True, dtype=np.uint16, voxel_size=voxel_size),
                        gt_mask: gp.ArraySpec(interpolatable=False, dtype=np.bool, voxel_size=voxel_size),
                    }
                ) +
                Convert(gt_mask, np.uint8) +
                gp.Pad(raw, context) +
                gp.Pad(gt_mask, context) +
                gp.RandomLocation()
                for sample in f)

    pipeline = (
            data_sources +
            gp.RandomProvider() +
            gp.Reject(gt_mask, min_masked=0.005, reject_probability=0.98) +
            nl.Clip(raw, 0, clip_max) +
            gp.ElasticAugment(
                control_point_spacing=[20, 20, 20],
                jitter_sigma=[1, 1, 1],
                rotation_interval=[0, math.pi/2.0],
                subsample=4) +
            gp.SimpleAugment(mirror_only=[1,2], transpose_only=[1,2]) +
            gp.Normalize(raw, factor=1.0 / clip_max) +
            #gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1) +
            #gp.IntensityScaleShift(raw, 2,-1) +
            nl.ConvertRgbToHlsVector(raw, hsl) +

            gp.BalanceLabels(gt_mask, loss_weights) +

            # train
            gp.PreCache(
                cache_size=40,
                num_workers=5) +
            gp.tensorflow.Train(
                os.path.join(output_folder, name),
                optimizer=net_names['optimizer'],
                loss=net_names['loss'],
                inputs={
                    net_names['raw']: hsl,
                    net_names['gt']: gt_mask,
                    net_names['loss_weights']: loss_weights,
                },
                outputs={
                    net_names['pred']: pred_mask,
                },
                gradients={
                    net_names['pred']: loss_gradient,
                },
                save_every=5000) +

            # visualize
            gp.Snapshot({
                    raw: 'volumes/raw',
                    pred_mask: 'volumes/pred_mask',
                    gt_mask: 'volumes/gt_mask',
                    hsl: 'volumes/hsl',
                    loss_weights: 'volumes/loss_weights',
                    loss_gradient: 'volumes/gradient',
                },
                output_filename=os.path.join(output_folder, 'snapshots', 'batch_{iteration}.hdf'),
                additional_request=snapshot_request,
                every=500) +
            gp.PrintProfilingStats(every=500)
    )

    with gp.build(pipeline):
        
        print("Starting training...")
        for i in range(max_iteration - trained_until):
            pipeline.request_batch(request)

if __name__ == "__main__":

    iteration = 200000
    root = '/groups/kainmueller/home/maisl/workspace/neurolight/experiments'
    experiment = 'setup03_00030419_1000'
    name = 'train_net'
    clip_max = 2000

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

    train_until(iteration, name, output_folder, clip_max)

