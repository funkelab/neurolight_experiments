import gunpowder as gp
import numpy as np
import neurolight.gunpowder as nl
import h5py
import sys
import glob
import os
import json
import logging
from datetime import datetime
import zarr


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


def predict(data_dir,
            train_dir,
            iteration,
            sample,
            test_net_name='train_net',
            train_net_name='train_net',
            output_dir='.',
            clip_max=1000
            ):

    if "hdf" not in data_dir:
        return

    print("Predicting ", sample)
    print('checkpoint: ', os.path.join(train_dir, train_net_name + '_checkpoint_%d' % iteration))

    checkpoint = os.path.join(train_dir, train_net_name + '_checkpoint_%d' % iteration)

    with open(os.path.join(train_dir, test_net_name + '_config.json'), 'r') as f:
        net_config = json.load(f)

    with open(os.path.join(train_dir, test_net_name + '_names.json'), 'r') as f:
        net_names = json.load(f)

    # ArrayKeys
    raw = gp.ArrayKey('RAW')
    pred_mask = gp.ArrayKey('PRED_MASK')

    input_shape = gp.Coordinate(net_config['input_shape'])
    output_shape = gp.Coordinate(net_config['output_shape'])

    voxel_size = gp.Coordinate((1, 1, 1))
    context = gp.Coordinate(input_shape - output_shape) / 2

    # add ArrayKeys to batch request
    request = gp.BatchRequest()
    request.add(raw, input_shape, voxel_size=voxel_size)
    request.add(pred_mask, output_shape, voxel_size=voxel_size)

    print("chunk request %s" % request)

    source = (
        gp.Hdf5Source(
            data_dir,
            datasets={
                raw: sample + '/raw',
            },
            array_specs={
                raw: gp.ArraySpec(interpolatable=True, dtype=np.uint16, voxel_size=voxel_size),
            },
        ) +
        gp.Pad(raw, context) +
        nl.Clip(raw, 0, clip_max) +
        gp.Normalize(raw, factor=1.0/clip_max) +
	    gp.IntensityScaleShift(raw, 2,-1)
    )

    with gp.build(source):
        raw_roi = source.spec[raw].roi
        print("raw_roi: %s" % raw_roi)
        sample_shape = raw_roi.grow(-context, -context).get_shape()

    print(sample_shape)

    # create zarr file with corresponding chunk size
    zf = zarr.open(os.path.join(output_dir, sample + '.zarr'), mode='w')

    zf.create('volumes/pred_mask', shape=sample_shape, chunks=output_shape, dtype=np.float16)
    zf['volumes/pred_mask'].attrs['offset'] = [0, 0, 0]
    zf['volumes/pred_mask'].attrs['resolution'] = [1, 1, 1]

    pipeline = (
        source +
        gp.tensorflow.Predict(
            graph=os.path.join(train_dir, test_net_name + '.meta'),
            checkpoint=checkpoint,
            inputs={
                net_names['raw']: raw,
            },
            outputs={
                net_names['pred']: pred_mask,
            },
            array_specs={
                pred_mask: gp.ArraySpec(
                    roi=raw_roi.grow(-context, -context),
                    voxel_size=voxel_size),
            },
            max_shared_memory=1024*1024*1024
            ) +
        Convert(pred_mask, np.float16) +
        gp.ZarrWrite(
            dataset_names={
                pred_mask: 'volumes/pred_mask',
            },
            output_dir=output_dir,
            output_filename=sample + '.zarr',
            compression_type='gzip',
            dataset_dtypes={
                pred_mask: np.float16}
        ) +

        # show a summary of time spend in each node every x iterations
        gp.PrintProfilingStats(every=100) +
        gp.Scan(reference=request, num_workers=5, cache_size=50)
    )

    with gp.build(pipeline):

        pipeline.request_batch(gp.BatchRequest())


if __name__ == "__main__":

    data_dir = "/groups/kainmueller/home/maisl/workspace/patch_instance_segmentation/patch_instance_segmentation/01_data/flylight_270319_val.hdf"
    #sample = "BJD_118D12_AE_01-20171020_66_D2"
    root = '/groups/kainmueller/home/maisl/workspace/neurolight/experiments'
    experiment = 'setup01_030419_01'
    iteration = 100000
    test_net_name = 'test_net'
    train_net_name = 'train_net'
    clip_max = 1000
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if len(sys.argv) > 1:
        experiment = sys.argv[1]
        iteration = int(sys.argv[2])
    
    if len(sys.argv) > 3:
        clip_max = int(sys.argv[3])

    if len(sys.argv) > 4:
        sample = sys.argv[4]

    print("arguments: ", experiment, iteration, clip_max, sample)

    output_dir = os.path.join(root, experiment, 'test', '%d' % iteration)
    train_dir = os.path.join(root, experiment, 'train')
    try:
        os.makedirs(output_dir)
    except:
        pass

    logging.basicConfig(level=logging.INFO)

    start = datetime.now()

    print('call: ', data_dir, train_dir, iteration, sample, test_net_name, train_net_name, output_dir, clip_max)

    predict(
        data_dir, train_dir, iteration, sample,
        test_net_name=test_net_name,
        train_net_name=train_net_name,
        output_dir=output_dir, clip_max=clip_max)

    print(datetime.now() - start)
