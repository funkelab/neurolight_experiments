import neuroglancer
import h5py
import daisy
import os
import natsort
import numpy as np
import sys

neuroglancer.set_server_bind_address('0.0.0.0')


def add(s, a, name, shader=None, normalize=False, offset=None, voxel_size=None):

    if type(a) == daisy.array.Array:
        data = a.to_ndarray()
    else:
        data = a

    if normalize:
        if np.min(data) < 0:
            data += np.abs(np.min(data))
        data = data.astype(np.float32) / np.max(data)

    if shader == 'rgb':
        shader = """void main() { 
                emitRGB(vec3(toNormalized(getDataValue(0)), toNormalized(getDataValue(1)), toNormalized(getDataValue(2)))); 
                }"""

    if shader == 'gray':
        shader = """void main() {
                emitGrayscale(toNormalized(getDataValue(0)));
                }"""

    if shader == 'red':
        shader = """void main() {
                float value = toNormalized(getDataValue(0));
                if (value <= 0.0) color = vec4(0,0,0,0);
                if (value > 0.0) color = vec4(1,0,0,1);
                emitRGBA(color);
                }"""

    if shader == 'green':
        shader = """void main() {
                float value = toNormalized(getDataValue(0));
                if (value <= 0.0) color = vec4(0,0,0,0);
                if (value > 0.0) color = vec4(0.13, 0.48, 0.6, 1);
                emitRGBA(color);
                }"""

    kwargs = {}
    if shader is not None:
        kwargs['shader'] = shader

    if offset is None:
        offset = a.roi.get_offset()[::-1]

    if voxel_size is None:
        voxel_size = a.voxel_size[::-1]

    s.layers.append(
        name=name,
        layer=neuroglancer.LocalVolume(
            data=data,
            offset=offset,
            voxel_size=voxel_size
        ),
        **kwargs)

def two_color_coded(a):

    if type(a) == daisy.array.Array:
        data = a.to_ndarray()
    else:
        data = a

    pos = np.zeros_like(data)
    neg = np.zeros_like(data)
    zero = np.zeros_like(data)

    idx = data > 0
    if np.sum(idx) > 0:
        pos[idx] = data[idx] #/ float(np.max(data[idx]))
    idx = data < 0
    if np.sum(idx) > 0:
        neg[idx] = data[idx] * (-1) #/ float(np.min(data[idx]))

    color = np.stack([pos, neg, zero], axis=0)

    return color

root = '/groups/kainmueller/home/maisl/workspace/neurolight/experiments'
experiments = [  # 'setup04_070419_00',
    # 'setup02_030419_00',
    # 'setup05_070419_01',
    # 'setup04_00050419',
    # 'setup05_01050419',
    # 'setup06_070419_00',
    # 'setup07_070419_00',
    # 'setup08_080419_00',
    # 'setup09_090419_00',
    'setup10_090419_00']
expid = 0
experiment = experiments[expid]
sn = 'snapshots'

sn_path = os.path.join(root, experiment, 'train', sn)
snapshots = os.listdir(sn_path)
snapshots = natsort.natsorted(snapshots)
snapshot = snapshots[-1]
print(sn_path, snapshot)
snapshot = 'batch_12001.hdf'

f = os.path.join(sn_path, snapshot)
raw = daisy.open_ds(f, 'volumes/raw')
print(raw.roi.get_shape(), raw.to_ndarray().shape)

gt_mask = daisy.open_ds(f, 'volumes/gt_mask')
gt_dt = daisy.open_ds(f, 'volumes/gt_dt')
pred_dt = daisy.open_ds(f, 'volumes/pred_dt')
gradient = daisy.open_ds(f, 'volumes/gradient')

viewer = neuroglancer.Viewer()

print(raw.roi.get_offset(),raw.roi.get_offset()[::-1])
print(gt_mask.roi.get_offset(),gt_mask.roi.get_offset()[::-1])



with viewer.txn() as s:

    add(s, raw, 'raw', shader='rgb', normalize=True)
    #add(s, gt_mask, 'gt_mask')
    add(s,
        two_color_coded(gt_dt),
        'gt_dt',
        shader='rgb',
        offset=gt_dt.roi.get_offset()[::-1],
        voxel_size=gt_dt.voxel_size[::-1])

    add(s,
        two_color_coded(pred_dt),
        'pred_dt',
        shader='rgb',
        offset=pred_dt.roi.get_offset()[::-1],
        voxel_size=pred_dt.voxel_size[::-1])

    add(s,
        two_color_coded(gradient),
        'gradient',
        shader='rgb',
        offset=gradient.roi.get_offset()[::-1],
        voxel_size=gradient.voxel_size[::-1])

print(viewer)
#pdb.set_trace()
print('exit')