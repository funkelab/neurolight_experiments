import numpy as np
import gunpowder as gp

from neurolight.gunpowder.nodes.grow_labels import GrowLabels
from neurolight.gunpowder.nodes.rasterize_skeleton import RasterizeSkeleton
from neurolight.gunpowder.nodes.topological_graph_matching import TopologicalMatcher
from neurolight.gunpowder.nodes.graph_source import GraphSource
from gunpowder import BatchRequest, build, Coordinate

import math
import copy
import time
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.DEBUG)
logging.getLogger(gp.nodes.random_location.__name__).setLevel(logging.INFO)

# Get snapshot of data pre GetNeuronPair

SEP_DIST = int(sys.argv[1])
SEPERATE_DISTANCE = [SEP_DIST * 0.7, SEP_DIST * 1.3]


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
        spec.dtype = np.int32

        binarized = gp.Array(data=(batch[self.gt].data > 0).astype(np.int32), spec=spec)

        batch[self.gt_binary] = binarized


class Crop(gp.BatchFilter):
    def __init__(self, input_array: gp.ArrayKey, output_array: gp.ArrayKey):

        self.input_array = input_array
        self.output_array = output_array

    def setup(self):

        spec = self.spec[self.input_array].copy()
        self.provides(self.output_array, spec)

    def prepare(self, request):
        pass

    def process(self, batch, request):
        input_data = batch[self.input_array].data
        input_spec = batch[self.input_array].spec
        input_roi = input_spec.roi
        output_roi_shape = request[self.output_array].roi.get_shape()
        shift = (input_roi.get_shape() - output_roi_shape) / 2
        output_roi = gp.Roi(shift, output_roi_shape)
        output_data = input_data[
            tuple(
                map(
                    slice,
                    output_roi.get_begin() / input_spec.voxel_size,
                    output_roi.get_end() / input_spec.voxel_size,
                )
            )
        ]
        output_spec = copy.deepcopy(input_spec)
        output_spec.roi = output_roi

        output_array = gp.Array(output_data, output_spec)

        batch[self.output_array] = output_array


input_size = Coordinate([74, 260, 260])
output_size = Coordinate([42, 168, 168])
path_to_data = Path("/nrs/funke/mouselight-v2")

# array keys for data sources
raw = gp.ArrayKey("RAW")
consensus = gp.PointsKey("CONSENSUS")
skeletonization = gp.PointsKey("SKELETONIZATION")
matched = gp.PointsKey("MATCHED")
labels = gp.ArrayKey("LABELS")

# output data
labels_fg = gp.ArrayKey("LABELS_FG")
labels_fg_bin = gp.ArrayKey("LABELS_FG_BIN")

loss_weights = gp.ArrayKey("LOSS_WEIGHTS")

voxel_size = gp.Coordinate((10, 3, 3))
input_size = input_size * voxel_size
output_size = output_size * voxel_size

# add request

# add snapshot request
# request.add(fg, output_size)
# request.add(labels_fg, output_size)
# request.add(gradient_fg, output_size)

data_sources = tuple(
    (
        gp.N5Source(
            filename=str(
                (
                    filename
                    / "consensus-neurons-with-machine-centerpoints-labelled-as-swcs-carved.n5"
                ).absolute()
            ),
            datasets={raw: "volume"},
            array_specs={
                raw: gp.ArraySpec(
                    interpolatable=True, voxel_size=voxel_size, dtype=np.uint16
                )
            },
        ),
        GraphSource(
            filename=str(
                Path(
                    "/groups/mousebrainmicro/home/pattonw/Code/Packages/neurolight/visualizations/skeletonization_carved-002-100.obj"
                ).absolute()
            ),
            points=(skeletonization,),
            scale=voxel_size,
            transpose=(2, 1, 0),
        ),
        GraphSource(
            filename=str(
                Path(
                    "/groups/mousebrainmicro/home/pattonw/Code/Packages/neurolight/visualizations/consensus-002.obj"
                ).absolute()
            ),
            points=(consensus,),
            scale=voxel_size,
            transpose=(2, 1, 0),
        ),
    )
    + gp.MergeProvider()
    + gp.RandomLocation(
        ensure_nonempty=consensus, ensure_centered=True, point_balance_radius=700
    )
    + TopologicalMatcher(skeletonization, consensus, matched)
    + RasterizeSkeleton(
        points=matched,
        array=labels,
        array_spec=gp.ArraySpec(
            interpolatable=False, voxel_size=voxel_size, dtype=np.uint32
        ),
    )
    + GrowLabels(labels, radii=[10])
    # augment
    + gp.ElasticAugment(
        [40, 10, 10],
        [0.25, 1, 1],
        [0, math.pi / 2.0],
        subsample=4,
        use_fast_points_transform=True,
        recompute_missing_points=False,
    )
    + gp.SimpleAugment(mirror_only=[1, 2], transpose_only=[1, 2])
    + gp.Normalize(raw)
    + gp.IntensityAugment(raw, 0.9, 1.1, -0.001, 0.001)
    for filename in path_to_data.iterdir()
    if "2018-07-02" in filename.name
)

pipeline = (
    data_sources
    + gp.RandomProvider()
    + Crop(labels, labels_fg)
    + BinarizeGt(labels_fg, labels_fg_bin)
    + gp.BalanceLabels(labels_fg_bin, loss_weights)
    + gp.PrintProfilingStats(every=10)
    + gp.Snapshot(
        output_filename="snapshot_NA_{}.hdf".format(SEP_DIST, "{iteration}"),
        dataset_names={
            raw: "volumes/raw",
            labels: "volumes/labels",
            labels_fg_bin: "volumes/labels_fg_bin",
        },
        every=1,
    )
)

request = BatchRequest()

# add request
request = gp.BatchRequest()
request.add(raw, input_size)
# don't need this, but random location assumes it is provided
request.add(matched, input_size)
request.add(labels_fg, output_size)
request.add(labels_fg_bin, output_size)
request.add(loss_weights, output_size)
request.add(labels, input_size)

# add snapshot request
# request.add(fg, output_size)
# request.add(labels_fg, output_size)
# request.add(gradient_fg, output_size)

with build(pipeline):
    t1 = time.time()
    for _ in range(10):
        pipeline.request_batch(request)
