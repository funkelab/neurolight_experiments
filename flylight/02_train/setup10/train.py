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
from scipy import ndimage


data_dir = '/groups/kainmueller/home/maisl/workspace/patch_instance_segmentation/patch_instance_segmentation/01_data'
data_files = [
    'flylight_060419_train.hdf',
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


class DistanceTransform(gp.BatchFilter):

    def __init__(self, array, dt, scaling):

        self.array = array
        self.dt = dt
        self.scaling = scaling

    def setup(self):

        spec = self.spec[self.array].copy()
        spec.dtype = np.int32
        self.provides(self.dt, spec)

    def process(self, batch, request):
        
        mask = batch[self.array].data.copy() > 0
        spec = batch[self.array].spec.copy()
        spec.dtype = np.float32

        bg = np.logical_not(mask)
        dt = np.tanh(ndimage.distance_transform_edt(mask) / self.scaling)
        dt[bg] = np.tanh(ndimage.distance_transform_edt(bg) / self.scaling)[bg] * (-1)

        batch[self.dt] = gp.Array(data=dt.astype(np.float32), spec=spec)


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
    gt_mask = gp.ArrayKey('GT_MASK')
    gt_dt = gp.ArrayKey('GT_DT')
    pred_dt = gp.ArrayKey('PRED_DT')
    loss_gradient = gp.ArrayKey('LOSS_GRADIENT')

    voxel_size = gp.Coordinate((1, 1, 1))
    input_shape = gp.Coordinate(net_config['input_shape'])
    output_shape = gp.Coordinate(net_config['output_shape'])
    context = gp.Coordinate(input_shape - output_shape) / 2

    request = gp.BatchRequest()
    request.add(raw, input_shape)
    request.add(gt_mask, output_shape)
    request.add(gt_dt, output_shape)

    snapshot_request = gp.BatchRequest()
    snapshot_request.add(raw, input_shape)
    snapshot_request.add(gt_mask, output_shape)
    snapshot_request.add(gt_dt, output_shape)
    snapshot_request.add(pred_dt, output_shape)
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
            gp.Reject(gt_mask, min_masked=0.005, reject_probability=1.) +
            DistanceTransform(gt_mask, gt_dt, 3) +
            nl.Clip(raw, 0, clip_max) +
            gp.Normalize(raw, factor=1.0/clip_max) +
            gp.ElasticAugment(
                control_point_spacing=[20, 20, 20],
                jitter_sigma=[1, 1, 1],
                rotation_interval=[0, math.pi/2.0],
                subsample=4) +
            gp.SimpleAugment(mirror_only=[1,2], transpose_only=[1,2]) +

            gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1) +
            gp.IntensityScaleShift(raw, 2,-1) +

            # train
            gp.PreCache(
                cache_size=40,
                num_workers=5) +
            gp.tensorflow.Train(
                os.path.join(output_folder, name),
                optimizer=net_names['optimizer'],
                loss=net_names['loss'],
                inputs={
                    net_names['raw']: raw,
                    net_names['gt_dt']: gt_dt,
                },
                outputs={
                    net_names['pred_dt']: pred_dt,
                },
                gradients={
                    net_names['pred_dt']: loss_gradient,
                },
                save_every=5000) +

            # visualize
            gp.Snapshot({
                    raw: 'volumes/raw',
                    gt_mask: 'volumes/gt_mask',
                    gt_dt: 'volumes/gt_dt',
                    pred_dt: 'volumes/pred_dt',
                    loss_gradient: 'volumes/gradient',
                },
                output_filename=os.path.join(output_folder, 'snapshots', 'batch_{iteration}.hdf'),
                additional_request=snapshot_request,
                every=2000) +
            gp.PrintProfilingStats(every=500)
    )

    with gp.build(pipeline):
        
        print("Starting training...")
        for i in range(max_iteration - trained_until):
            pipeline.request_batch(request)


if __name__ == "__main__":

    iteration = 100000
    root = '/groups/kainmueller/home/maisl/workspace/neurolight/experiments'
    experiment = 'setup10_090419_00'
    name = 'train_net'
    clip_max = 1500

    #os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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
    
    print('Start training with: ', experiment, iteration, name, clip_max, output_folder)
    train_until(iteration, name, output_folder, clip_max)

