import numpy as np
import zarr
import h5py
import voi
import rand
import iou
import sys
import glob
import os
from skimage import io, morphology, measure
import glob
import csv
#from matplotlib import pyplot as plt
from datetime import datetime


def evaluate(pred_folder, data_folder, dataset, output_folder):

    thresholds = [0.9, 0.95, 0.99]

    hf = h5py.File(os.path.join(data_folder, dataset + '.hdf'), 'r')
    ds_samples = list(hf.keys())
    pred_samples = glob.glob(os.path.join(pred_folder, "*.zarr"))
    print(pred_samples)
    print('output folder: ', output_folder)

    cf = open(os.path.join(output_folder, 'iou.csv'), 'w')
    writer = csv.writer(cf, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['sample', ] + thresholds)
    iou_results = {}

    for sample in pred_samples:

        name = os.path.basename(sample)[:-5]

        if name not in ds_samples:
            print(name, " not in gt dataset! Skipping...")
            continue

        gt = np.asarray(hf[name + '/fg'], dtype=np.uint8)

        zf = zarr.open(sample)
        pred = np.asarray(zf['volumes/pred_mask'])

        mip = np.max(gt, axis=0)
        mip[mip > 0] = 255
        io.imsave(os.path.join(output_folder, name + '_raw.png'), mip)

        results = []

        for thresh in thresholds:

            print('sample: ', name)

            mask = (pred >= thresh).astype(np.uint8)
            mask = morphology.remove_small_objects(
                measure.label(mask, background=0, connectivity=1),
                min_size=20, connectivity=1)
            mask = (mask > 0).astype(np.uint8)

            #result = voi.voi(mask, gt)
            #print('threshold: ', thresh, ', result: ', result)

            #result = rand.adapted_rand(mask, gt)
            #print('adapted rand for threshold: ', thresh, ', result: ', result)

            result = iou.intersection_over_union(np.expand_dims(mask, -1), gt)
            print('iou for threshold: ', thresh, ', result: ', result)
            results.append(result[1])

            mip = np.max(mask, axis=0)
            mip[mip > 0] = 255
            io.imsave(os.path.join(output_folder, name + '_mask_' + str(thresh) + '.png'), mip)

        iou_results[name] = results
        print(len(iou_results))

    avg = None
    for k,v in iou_results.items():
        writer.writerow([k, ] + v)
        if avg is None:
            avg = np.asarray(v)
        else:
            avg += np.asarray(v)

    avg /= len(iou_results)
    writer.writerow(['Average', ] + list(avg))


if __name__ == "__main__":

    data_folder = "/groups/kainmueller/home/maisl/workspace/patch_instance_segmentation/patch_instance_segmentation/01_data"
    dataset = "flylight_270319_val"

    root = '/groups/kainmueller/home/maisl/workspace/neurolight/experiments'
    experiment = 'setup02_030419_00'
    iteration = 100000

    if len(sys.argv) > 1:
        experiment = sys.argv[1]
        iteration = int(sys.argv[2])

    if len(sys.argv) > 3:
        dataset = sys.argv[3]

    pred_folder = os.path.join(root, experiment, 'test', '%d' % iteration)
    output_folder = os.path.join(root, experiment, 'eval', '%d' % iteration)

    try:
        os.makedirs(output_folder)
    except OSError:
        pass

    evaluate(pred_folder, data_folder, dataset, output_folder)



