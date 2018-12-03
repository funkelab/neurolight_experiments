from __future__ import print_function
import gunpowder as gp
from gunpowder.roi import Roi
import numpy as np
import json
import logging
import os
import tensorflow as tf
import h5py
import sys
import math

data_dir = '../01_data'
data_files = [
        'flylight_test_split.hdf',
        #'perfect_neurons.hdf',
        #'neurons_kaiyu.hdf',
        ]


class MergeColorChannel(gp.BatchFilter):

    def __init__(self, red, green, blue, raw):
        
        self.red = red
        self.green = green
        self.blue = blue
        self.raw = raw

    def setup(self):

        spec = self.spec[self.red].copy()
        spec.roi = Roi((0,) + spec.roi.get_offset(), (3,) + spec.roi.get_shape())
        self.provides(self.raw, spec)

    def prepare(self, request):
        pass

    def process(self, batch, request):

        spec = self.spec[self.red].copy()
        merged = np.stack([batch[self.red].data, batch[self.green].data, batch[self.blue].data],
                axis=0)
        batch[self.raw] = gp.Array(data=merged.astype(spec.dtype),
                                   spec=gp.ArraySpec(dtype=spec.dtype,
                                                     roi=Roi((0, 0, 0, 0), merged.shape),
                                                     interpolatable=True,
                                                     voxel_size=gp.Coordinate((1, 1, 1, 1))))


class BinarizeGt(gp.BatchFilter):

    def __init__(self, gt, gt_binary):

        self.gt = gt
        self.gt_binary = gt_binary

    def setup(self):

        spec = self.spec[self.gt].copy()
        spec.dtype = np.uint8
        self.provides(self.gt_binary, spec)

    def prepare(self, request):
        pass

    def process(self, batch, request):

        spec = batch[self.gt].spec.copy()
        spec.dtype = np.uint8

        binarized = gp.Array(
            data=(batch[self.gt].data > 0).astype(np.uint8),
            spec=spec)

        batch[self.gt_binary] = binarized


class RGBToHSLVector(gp.BatchFilter):
    
    def __init__(self, raw, hsl):

        self.raw = raw
        self.hsl = hsl

    def setup(self):

        spec = self.spec[self.raw].copy()
        spec.dtype = np.float32
        self.provides(self.hsl, spec)

    def prepare(self, request):
        pass

    def process(self, batch, request):

        spec = batch[self.raw].spec.copy()
        spec.dtype = np.float32

        rgb = batch[self.raw].data

        # c, z, y, x
        # convert rgb to hsl, assume that rgb is float [0,1]
        maxc = np.max(rgb, axis=0)
        minc = np.min(rgb, axis=0)
        dst = np.zeros_like(rgb, dtype=np.float32)

        dst[1] = (minc + maxc) / 2.0
        
        not_white = minc != maxc
        minus = maxc - minc
        plus = maxc + minc

        dark = dst[1] < 0.5
        idx = np.logical_and(dark, not_white)
        dst[2, idx] = np.divide(minus[idx], plus[idx])
        
        idx = np.logical_and(np.logical_not(dark), not_white)
        dst[2, idx] = np.divide(minus[idx], (2.0-plus)[idx])

        rc = np.divide((maxc-rgb[0]), minus)
        gc = np.divide((maxc-rgb[1]), minus)
        bc = np.divide((maxc-rgb[2]), minus)

        idx = np.logical_and(rgb[0] == maxc, not_white)
        dst[0, idx] = bc[idx] - gc[idx]
        idx = np.logical_and(rgb[1] == maxc, not_white)
        dst[0, idx] = 2.0+rc[idx]-bc[idx]
        idx = np.logical_and(rgb[2] == maxc, not_white)
        dst[0, idx] = (4.0+gc-rc)[idx]
        
        dst[0] = (dst[0]/6.0) % 1.0

        # convert hsl to vector
        dst[1] = dst[1] * np.sin(np.radians(dst[0]))
        dst[0] = np.cos(np.radians(dst[0]))

        batch[self.hsl] = gp.Array(data=dst.astype(np.float32), spec=spec)


def train_until(max_iteration):

    # get the latest checkpoint
    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
    else:
        trained_until = 0
        if trained_until >= max_iteration:
            return

    with open('train_net_config.json', 'r') as f:
        net_config = json.load(f)
    with open('train_net_names.json', 'r') as f:
        net_names = json.load(f)
    
    raw = gp.ArrayKey('RAW')
    red = gp.ArrayKey('RED')
    green = gp.ArrayKey('GREEN')
    blue = gp.ArrayKey('BLUE')
    gt = gp.ArrayKey('GT')
    gt_binary = gp.ArrayKey('GT_BINARY')
    loss_weights = gp.ArrayKey('LOSS_WEIGHTS')
    prediction = gp.ArrayKey('PREDICTION')
    loss_gradient = gp.ArrayKey('LOSS_GRADIENT')
    hsl = gp.ArrayKey('HSL')
    
    voxel_size = gp.Coordinate((1, 1, 1))
    input_size = gp.Coordinate(net_config['input_shape'])
    output_size = gp.Coordinate(net_config['output_shape'])

    request = gp.BatchRequest()
    request.add(red, input_size)
    request.add(green, input_size)
    request.add(blue, input_size)
    request.add(gt, output_size)
    request.add(loss_weights, output_size)
    
    snapshot_request = gp.BatchRequest({
        prediction: request[gt],
        loss_gradient: request[gt],
        })

    data_sources = tuple()
    for data_file in data_files:
        current_path = os.path.join(data_dir, data_file)
        with h5py.File(current_path, 'r') as f:
            data_sources += tuple(
                    gp.Hdf5Source(
                        current_path,
                        datasets = {
                            red: sample + '/raw/red',
                            green: sample + '/raw/green',
                            blue: sample + '/raw/blue',
                            gt: sample + '/gt/ids'
                        },
                        array_specs = {
                            red: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
                            green: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
                            blue: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
                            gt: gp.ArraySpec(interpolatable=False, voxel_size=voxel_size)
                            }
                    ) +
                    gp.RandomLocation()
                    for sample in f)

    pipeline = (
            data_sources +
            gp.RandomProvider() +

            # augment --> todo: add some color permutation / augmentation ...
            gp.ElasticAugment(
                control_point_spacing=[10, 10, 10],
                jitter_sigma=[1, 1, 1],
                rotation_interval=[0, math.pi / 2.0],
                subsample=8) +
            gp.SimpleAugment() +

            MergeColorChannel(red, green, blue, raw) +
            gp.Normalize(raw) +
            gp.IntensityAugment(raw, 0.9, 1.1, -0.001, 0.001) +

            BinarizeGt(gt, gt_binary) +
            gp.BalanceLabels(gt_binary, loss_weights) +
            
            RGBToHSLVector(raw, hsl) +
            # train
            gp.PreCache(
                cache_size=40,
                num_workers=10) +
            gp.tensorflow.Train(
                './train_net',
                optimizer=net_names['optimizer'],
                loss=net_names['loss'],
                inputs={
                    net_names['raw']: hsl,
                    net_names['gt']: gt_binary,
                    net_names['loss_weights']: loss_weights,
                },
                outputs={
                    net_names['prediction']: prediction,
                },
                gradients={
                    net_names['prediction']: loss_gradient,
                },
                save_every=50000) +

            # visualize
            gp.Snapshot({
                    raw: 'volumes/raw',
                    prediction: 'volumes/prediction',
                    gt: 'volumes/gt',
                    gt_binary: 'volumes/gt_binary',
                    hsl: 'volumes/hsl',
                    loss_weights: 'volumes/loss_weights',
                    loss_gradient: 'volumes/gradient',
                },
                output_filename='snapshot_{iteration}.hdf',
                additional_request=snapshot_request,
                every=10) +
            gp.PrintProfilingStats(every=10)
    )

    with gp.build(pipeline):
        
        print("Starting training...")
        for i in range(max_iteration - trained_until):
            pipeline.request_batch(request)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    iteration = int(sys.argv[1])
    train_until(iteration)

