import numpy as np
import h5py
import os

input_file = 'flylight_test.hdf'
output_file = 'flylight_test_split.hdf'

if os.path.exists(output_file):
    os.remove(output_file)

infile = h5py.File(input_file, 'r')

for sample in infile:
    raw = np.array(infile[sample + '/raw'])
    gt = np.array(infile[sample + '/gt/neuron_ids'])

    with h5py.File(output_file, 'a') as outfile:
        vol = outfile.create_group(sample)
        vol.create_dataset(
            'raw/red',
            data=raw[0],
            compression='gzip')
        vol.create_dataset(
            'raw/green',
            data=raw[1],
            compression='gzip')
        vol.create_dataset(
            'raw/blue',
            data=raw[2],
            compression='gzip')
        vol.create_dataset(
            'gt/ids',
            data=gt,
            compression='gzip')