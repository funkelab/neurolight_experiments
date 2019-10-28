from __future__ import print_function

import numpy as np
import json
import logging
import tensorflow as tf
import sys
import math
from pathlib import Path
import networkx as nx
import copy

import gunpowder as gp

from neurolight.gunpowder import FusionAugment, RasterizeSkeleton
from neurolight.gunpowder.nodes.graph_source import GraphSource
from neurolight.gunpowder.nodes.topological_graph_matching import TopologicalMatcher
from neurolight.gunpowder.nodes.get_neuron_pair import GetNeuronPair

# from neurolight.gunpowder.recenter import Recenter
from neurolight.gunpowder.nodes.grow_labels import GrowLabels

sample_dir = "/nrs/funke/mouselight-v2"


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
        print(input_roi, output_roi)
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


with open("train_net_config.json", "r") as f:
    net_config = json.load(f)
with open("train_net_names.json", "r") as f:
    net_names = json.load(f)


def train_until(max_iteration):
    SEPERATE_BY = (150 * 0.9, 150 * 1.1)

    # get the latest checkpoint
    if tf.train.latest_checkpoint("."):
        trained_until = int(tf.train.latest_checkpoint(".").split("_")[-1])
    else:
        trained_until = 0
        if trained_until >= max_iteration:
            return

    # array keys for data sources
    raw = gp.ArrayKey("RAW")
    consensus = gp.PointsKey("CONSENSUS")
    skeletonization = gp.PointsKey("SKELETONIZATION")
    nonempty_placeholder = gp.PointsKey("PLACEHOLDER")
    matched = gp.PointsKey("MATCHED")
    labels = gp.ArrayKey("LABELS")

    # array keys for base volume
    raw_base = gp.ArrayKey("RAW_BASE")
    labels_base = gp.ArrayKey("LABELS_BASE")
    matched_base = gp.PointsKey("SWC_BASE")

    # array keys for add volume
    raw_add = gp.ArrayKey("RAW_ADD")
    labels_add = gp.ArrayKey("LABELS_ADD")
    matched_add = gp.PointsKey("SWC_ADD")

    # array keys for fused volume
    raw_fused = gp.ArrayKey("RAW_FUSED")
    labels_fused = gp.ArrayKey("LABELS_FUSED")
    matched_fused = gp.PointsKey("SWC_FUSED")

    # output data
    labels_fg = gp.ArrayKey("LABELS_FG")
    labels_fg_bin = gp.ArrayKey("LABELS_FG_BIN")

    voxel_size = gp.Coordinate((10, 3, 3))
    input_size = gp.Coordinate(net_config["input_shape"]) * voxel_size
    output_size = gp.Coordinate(net_config["output_shape"]) * voxel_size

    # add request
    request = gp.BatchRequest()
    request.add(raw_fused, input_size)
    request.add(labels_fused, input_size)
    request.add(matched_fused, input_size)
    request.add(labels_fg, output_size)
    request.add(labels_fg_bin, output_size)

    # add snapshot request
    # request.add(fg, output_size)
    # request.add(labels_fg, output_size)
    request.add(raw_base, input_size)
    request.add(raw_add, input_size)
    request.add(labels_base, input_size)
    request.add(labels_add, input_size)
    request.add(matched_base, input_size)
    request.add(matched_add, input_size)

    full_skeletonization = Path(sample_dir) / "2018-07-02" / "skeletonization.obj"
    cutout_skeletonization = Path(
        "/groups/mousebrainmicro/home/pattonw/Code/Scripts/neurolight_experiments/mouselight/01_test/skeletonization_carved-002-100.obj"
    )
    all_consensus = Path(sample_dir) / "2018-07-02" / "skeletonization.obj"
    consensus_002 = Path(
        "/groups/mousebrainmicro/home/pattonw/Code/Scripts/neurolight_experiments/mouselight/01_test/consensus-002.obj"
    )

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
                filename=(consensus_002),
                points=(consensus, nonempty_placeholder),
                scale=voxel_size,
                transpose=(2, 1, 0),
            ),
            GraphSource(
                filename=(cutout_skeletonization),
                points=(skeletonization,),
                scale=voxel_size,
                transpose=(2, 1, 0),
            ),
        )
        + gp.MergeProvider()
        + gp.RandomLocation(ensure_nonempty=nonempty_placeholder, ensure_centered=True)
        + TopologicalMatcher(
            skeletonization, consensus, matched, 150, failures=Path("match_failures")
        )
        + RasterizeSkeleton(
            points=matched,
            array=labels,
            array_spec=gp.ArraySpec(
                interpolatable=False, voxel_size=voxel_size, dtype=np.uint32
            ),
        )
        + GrowLabels(labels, radius=10)
        # augment
        + gp.ElasticAugment([40, 10, 10], [0.25, 1, 1], [0, math.pi / 2.0], subsample=4)
        + gp.SimpleAugment(mirror_only=[1, 2], transpose_only=[1, 2])
        + gp.Normalize(raw)
        + gp.IntensityAugment(raw, 0.9, 1.1, -0.001, 0.001)
        for filename in Path(sample_dir).iterdir()
        if "2018-07-02" in filename.name
    )

    pipeline = (
        data_sources
        + gp.RandomProvider()
        + GetNeuronPair(
            matched,
            raw,
            labels,
            (matched_base, matched_add),
            (raw_base, raw_add),
            (labels_base, labels_add),
            seperate_by=SEPERATE_BY,
            shift_attempts=np.mean(SEPERATE_BY, dtype=int) * 2,
            request_attempts=10,
            nonempty_placeholder=nonempty_placeholder,
        )
        + FusionAugment(
            raw_base,
            raw_add,
            labels_base,
            labels_add,
            matched_base,
            matched_add,
            raw_fused,
            labels_fused,
            matched_fused,
            blend_mode="labels_mask",
            blend_smoothness=10,
            num_blended_objects=0,
        )
        + Crop(labels_fused, labels_fg)
        + BinarizeGt(labels_fg, labels_fg_bin)
        # + gp.BalanceLabels(labels_fg_bin, loss_weights)
        + gp.Snapshot(
            output_filename="snapshot_{iteration}_{id}.hdf",
            dataset_names={
                raw_fused: "volumes/raw_fused",
                raw_base: "volumes/raw_base",
                raw_add: "volumes/raw_add",
                labels_fused: "volumes/labels_fused",
                labels_base: "volumes/labels_base",
                labels_add: "volumes/labels_add",
                labels_fg_bin: "volumes/labels_fg_bin",
            },
            every=1,
        )
    )

    with gp.build(pipeline):

        logging.info("Starting training...")
        for i in range(max_iteration - trained_until):
            logging.info("requesting batch {}".format(i))
            batch = pipeline.request_batch(request)


def points_to_graph(points):
    g = nx.DiGraph()
    for point_id, point in points.items():
        g.add_node(point_id, location=point.location)
        if (
            point.parent_id is not None
            and point.parent_id != point_id
            and point.parent_id != -1
            and point.parent_id in points
        ):
            g.add_edge(point_id, point.parent_id)
    return g


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting training!")

    iteration = int(sys.argv[1])
    train_until(iteration)
