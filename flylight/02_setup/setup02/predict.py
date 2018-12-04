import sys

import gunpowder as gp
from gunpowder import *
from gunpowder.tensorflow import Predict
from maislight.gunpowder import Hdf5ChannelSource, MergeChannel

import os
import json
import numpy as np

raw_file = 'experiments/01_data/2017-09-25/2017-09-25_G-004_carved.h5'

def predict(iteration, sample, read_roi, out_file, write_roi):

    setup_dir = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(setup_dir, 'test_net_config.json'), 'r') as f:
        config = json.load(f)

    fg = ArrayKey('FG')
    bg = ArrayKey('BG')
    raw = ArrayKey('RAW')
    prediction = gp.ArrayKey('PREDICTION')

    voxel_size = Coordinate((1, 1, 1))
    input_size = Coordinate(config['input_shape'])*voxel_size
    output_size = Coordinate(config['output_shape'])*voxel_size

    chunk_request = BatchRequest()
    chunk_request.add(fg, input_size)
    chunk_request.add(bg, input_size)
    chunk_request.add(raw, input_size)
    chunk_request.add(prediction, output_size)

    data_source = (

        Hdf5ChannelSource(
            raw_file,
            datasets={
                fg: '/volume',
                bg: '/volume',
            },
            channel_ids={
                fg: 0,
                bg: 1,
            },
            data_format='channels_last',
            array_specs={
                fg: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
                bg: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
            }
        ) +
        MergeChannel(fg, bg, raw)
    )

    with build(data_source):
        raw_spec = data_source.spec[raw]

    pipeline = (
        data_source +
        Pad(raw, size=None) +
        Crop(raw, read_roi) +
        Normalize(raw) +
        Predict(
            os.path.join(setup_dir, 'train_net_checkpoint_%d'%iteration),
            graph=os.path.join(setup_dir, 'test_net.meta'),
            inputs = {
                config['raw']: raw,
            },
            outputs = {
                config['prediction']: prediction,
            },
            array_specs = {
                prediction: ArraySpec(
                    roi=raw_spec.roi,
                    voxel_size=raw_spec.voxel_size,
                    dtype=np.float32
                )
            },
            skip_empty=True) +
        ZarrWrite(
            dataset_names={
                prediction: 'volumes/prediction'
            },
            output_filename=out_file
        ) +
        PrintProfilingStats(every=100) +
        Scan(chunk_request, num_workers=10)
    )

    with build(pipeline):
        pipeline.request_batch(BatchRequest())

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('gunpowder.nodes.generic_predict').setLevel(logging.DEBUG)

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = json.load(f)

    read_roi = Roi(
        config['read_begin'],
        config['read_shape'])
    write_roi = Roi(
        config['write_begin'],
        config['write_shape'])

    predict(
        config['iteration'],
        config['sample'],
        read_roi,
        config['out_file'],
        write_roi)
