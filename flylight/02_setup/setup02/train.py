from __future__ import print_function
import gunpowder as gp
from gunpowder.roi import Roi
from maislight.gunpowder import Hdf5ChannelSource, SwcSource, RasterizeSwcSkeleton
import numpy as np
import json
import logging
import tensorflow as tf
import sys
import math
import os

raw_files = [
    #'/nrs/mouselight/SAMPLES/',
    #'../../01_data',experiments
    '../../01_data/2017-09-25/2017-09-25_G-003_carved.h5',
    '../../01_data/2017-09-25/2017-09-25_G-006_carved.h5',
    ]
gt_files = [
    'experiments/01_data/2017-09-25/2017-09-25_G-003_carved_segmented.h5',
    'experiments/01_data/2017-09-25/2017-09-25_G-006_carved_segmented.h5',
]
swc_files = [
    #'/groups/mousebrainmicro/mousebrainmicro/tracing_complete',
    #'../01_data',
    ]

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
        print('sum fg: ', np.sum(binarized.data))

        batch[self.gt_binary] = binarized


class MergeChannel(gp.BatchFilter):

    def __init__(self, fg, bg, raw):
        self.fg = fg
        self.bg = bg
        self.raw = raw

    def setup(self):
        spec = self.spec[self.fg].copy()
        spec.roi = Roi((0,) + spec.roi.get_offset(), (3,) + spec.roi.get_shape())
        self.provides(self.raw, spec)

    def prepare(self, request):
        pass

    def process(self, batch, request):
        spec = self.spec[self.fg].copy()
        voxel_size = (1,) + spec.voxel_size
        merged = np.stack([batch[self.fg].data, batch[self.bg].data], axis=0)
        batch[self.raw] = gp.Array(data=merged.astype(spec.dtype),
                                   spec=gp.ArraySpec(dtype=spec.dtype,
                                                     roi=Roi((0, 0, 0, 0), merged.shape) * voxel_size,
                                                     interpolatable=True,
                                                     voxel_size=voxel_size))


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
    fg = gp.ArrayKey('FG')
    bg = gp.ArrayKey('BG')
    gt = gp.ArrayKey('GT')
    swc = gp.PointsKey('SWC')
    skeleton = gp.ArrayKey('SKELETON')
    gt_binary = gp.ArrayKey('GT_BINARY')
    loss_weights = gp.ArrayKey('LOSS_WEIGHTS')
    prediction = gp.ArrayKey('PREDICTION')
    loss_gradient = gp.ArrayKey('LOSS_GRADIENT')
    
    voxel_size = gp.Coordinate((3, 3, 10))
    input_size = gp.Coordinate(net_config['input_shape']) * voxel_size
    output_size = gp.Coordinate(net_config['output_shape']) * voxel_size

    request = gp.BatchRequest()
    request.add(fg, input_size)
    request.add(bg, input_size)
    request.add(swc, output_size)
    #request.add(gt, output_size)
    request.add(skeleton, output_size)
    request.add(loss_weights, output_size)
    
    snapshot_request = gp.BatchRequest({
        prediction: request[skeleton],
        loss_gradient: request[skeleton],
        })

    data_sources = tuple()
    data_sources += tuple(

        (Hdf5ChannelSource(
            raw_file,
            datasets = {
                fg: '/volume',
                bg: '/volume',
            },
            channel_ids = {
                fg: 0,
                bg: 1,
            },
            data_format = 'channels_last',
            array_specs = {
                fg: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
                bg: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
                }
        ),
        #gp.Hdf5Source(
        #    gt_file,
        #    datasets={
        #        gt: '/segmentation/AC',
        #    },
        #    array_specs={
        #        gt: gp.ArraySpec(interpolatable=False, voxel_size=voxel_size),
        #    }
        #),
        SwcSource(
            raw_file,
            '/reconstruction',
            swc
        )) +
        gp.MergeProvider() +
        gp.RandomLocation(ensure_nonempty=swc) +
        RasterizeSwcSkeleton(swc, skeleton, gp.ArraySpec(interpolatable=False, voxel_size=voxel_size), distance=(90, 90, 300)) +
        gp.Reject(skeleton, 0.001)
        for raw_file in raw_files
    )

    pipeline = (
            data_sources +
            gp.RandomProvider() +

            # augment
            gp.ElasticAugment(
                control_point_spacing=[10, 10, 10],
                jitter_sigma=[1, 1, 1],
                rotation_interval=[0, math.pi / 2.0],
                subsample=8) +
            #gp.SimpleAugment() +

            MergeChannel(fg, bg, raw) +
            gp.Normalize(raw) +
            gp.IntensityAugment(raw, 0.9, 1.1, -0.001, 0.001) +

            #BinarizeGt(gt, gt_binary) +
            gp.BalanceLabels(skeleton, loss_weights) +

            # train
            #gp.PreCache(
            #    cache_size=40,
            #    num_workers=10) +
            gp.tensorflow.Train(
                './train_net',
                optimizer=net_names['optimizer'],
                loss=net_names['loss'],
                inputs={
                    net_names['raw']: raw,
                    net_names['gt']: skeleton,
                    net_names['loss_weights']: loss_weights,
                },
                outputs={
                    net_names['prediction']: prediction,
                },
                gradients={
                    net_names['prediction']: loss_gradient,
                },
                save_every=20) +

            # visualize
            gp.Snapshot({
                    fg: 'volumes/raw',
                    prediction: 'volumes/prediction',
                    #gt: 'volumes/gt',
                    #gt_binary: 'volumes/gt_binary',
                    skeleton: '/volumes/skeleton',
                    loss_weights: 'volumes/loss_weights',
                    loss_gradient: 'volumes/gradient',
                },
                output_filename='snapshot_{iteration}.hdf',
                additional_request=snapshot_request,
                every=30) +
            gp.PrintProfilingStats(every=30)
    )

    with gp.build(pipeline):
        
        print("Starting training...")
        for i in range(max_iteration - trained_until):
            pipeline.request_batch(request)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) <= 1:
        iteration = 20
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print(os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        iteration = int(sys.argv[1])
    train_until(iteration)

